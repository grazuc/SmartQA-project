import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
import typer
from rich.console import Console
from typing import Optional, List, Dict, Any

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# Transformers (for LLM pipeline)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

app = typer.Typer(help="Smart Document QA - Ask questions about your .txt documents")
console = Console()

def load_env():
    load_dotenv()

class DocumentProcessor:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        if chunk_size is None:
            chunk_size = int(os.getenv("CHUNK_SIZE", "300"))
        if chunk_overlap is None:
            chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "40"))
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_document(self, path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.suffix.lower() == '.txt':
            console.print(f"[yellow]Warning: Expected .txt file, got {path.suffix}[/yellow]")
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                raise ValueError("Document is empty")
            return content
        except UnicodeDecodeError:
            raise ValueError(f"Could not decode file as UTF-8: {path}")

    def chunk_document(self, text: str) -> List[Document]:
        docs = self.splitter.create_documents([text])
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "chunk_id": str(i),
                "source": "input_document"
            })
        return docs

class SmartQA:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        load_dotenv()
        self.embed_model = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
        self.llm_model = os.getenv("LLM_MODEL", "google/flan-t5-small")
        self.top_k = int(os.getenv("TOP_K", "3"))
        self.log_file = Path(os.getenv("LOG_FILE", "qa_history.jsonl"))
        self.processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)
        self.llm = self._load_llm()
        self.vectorstore = None
        self._setup_prompt()

    def _load_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.3
        )
        return HuggingFacePipeline(pipeline=pipe)

    def _setup_prompt(self):
        self.qa_prompt = PromptTemplate(
            template="""You are a helpful assistant. Using ONLY the context below, answer the user's question.

Context:
{context}

Question: {question}

Answer (max 200 words):""",
            input_variables=["context", "question"]
        )

    def index_document(self, file_path: Path) -> int:
        text = self.processor.load_document(file_path)
        docs = self.processor.chunk_document(text)
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        console.print(f"[green]✓ Indexed {len(docs)} chunks from {file_path.name}[/green]")
        return len(docs)

    def ask_question(self, question: str) -> Dict[str, Any]:
        if not self.vectorstore:
            raise ValueError("No document indexed. Please load a document first.")
        start_time = time.time()
        docs = self.vectorstore.similarity_search(question, k=self.top_k)
        context = "\n\n".join([f"[Chunk {d.metadata['chunk_id']}]: {d.page_content}"
                              for d in docs])
        qa_chain = self.qa_prompt | self.llm
        answer = qa_chain.invoke({"context": context, "question": question}).strip()
        citations = [d.metadata.get('chunk_id', 'unknown') for d in docs]
        latency = time.time() - start_time
        question_tokens = len(question.split())
        answer_tokens = len(answer.split())
        total_tokens = question_tokens + answer_tokens

        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question": question,
            "answer": answer,
            "citations": citations,
            "latency": latency,
            "token_usage": total_tokens,
            "model": {
                "embedding": self.embed_model,
                "llm": self.llm_model
            }
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return {
            "answer": answer,
            "citations": citations,
            "latency": latency,
            "token_usage": total_tokens
        }

@app.command()
def index(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to text file"),
    chunk_size: int = typer.Option(None, "--chunk-size", help="Chunk size for text splitting"),
    chunk_overlap: int = typer.Option(None, "--chunk-overlap", help="Chunk overlap for text splitting")
):
    qa = SmartQA(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    file_path = Path(input_path)
    n_chunks = qa.index_document(file_path)
    console.print(f"[green]Document indexed with {n_chunks} chunks.[/green]")

@app.command()
def ask(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to text file"),
    question: str = typer.Option(..., "--ask", "-q", help="Question to ask"),
    chunk_size: int = typer.Option(None, "--chunk-size", help="Chunk size for text splitting"),
    chunk_overlap: int = typer.Option(None, "--chunk-overlap", help="Chunk overlap for text splitting")
):
    qa = SmartQA(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    file_path = Path(input_path)
    qa.index_document(file_path)
    result = qa.ask_question(question)
    console.print(f"[bold green]Answer:[/bold green] {result['answer']}\n")
    console.print(f"[dim]Citations: chunks {', '.join(result['citations'])} | Latency: {result['latency']:.2f}s | Tokens: {result['token_usage']}[/dim]")

if __name__ == "__main__":
    app()
