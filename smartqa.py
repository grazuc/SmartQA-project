import os
import time
import json
import typer
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document

# HuggingFace imports
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# UI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
app = typer.Typer(help="Smart Document QA - Ask questions about your .txt documents")

@dataclass
class QAResult:
    """Result of a question-answer interaction"""
    answer: str
    citations: List[str]
    latency: float
    confidence: Optional[float] = None

class DocumentProcessor:
    """Handles document loading and chunking, you can change the chunk_size and overlap on the .env"""
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    def load_document(self, path: Path) -> str:
        """Load text from file with error handling"""
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
        """Split document into chunks using chunk_size"""
        docs = self.splitter.create_documents([text])
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "chunk_id": str(i),
                "source": "input_document"
            })
        return docs

class SmartQA:
    """Main QA system"""
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        load_dotenv()
        self.embed_model = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
        self.llm_model = os.getenv("LLM_MODEL", "google/flan-t5-small")
        self.top_k = int(os.getenv("TOP_K", "3"))
        self.log_file = Path(os.getenv("LOG_FILE", "qa_history.jsonl"))

        self._initialize_models()
        self._setup_prompt()

        if chunk_size is None:
            chunk_size = int(os.getenv("CHUNK_SIZE", "300"))
        if chunk_overlap is None:
            chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "40"))

        self.processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    def _initialize_models(self):
        """Initialize embedding and LLM models"""
        console.print(f"[cyan]Loading embedding model: {self.embed_model}[/cyan]")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)
        console.print(f"[cyan]Loading LLM: {self.llm_model}[/cyan]")
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
        self.llm = HuggingFacePipeline(pipeline=pipe)
    def _setup_prompt(self):
        """Setup the QA prompt template"""
        self.qa_prompt = PromptTemplate(
            template="""You are a helpful assistant. Using ONLY the context below, answer the user's question.

Context:
{context}

Question: {question}

Answer (max 200 words):""",
            input_variables=["context", "question"]
        )
    def index_document(self, file_path: Path) -> int:
        """Load and index a document"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Loading document...", total=None)
            text = self.processor.load_document(file_path)
            docs = self.processor.chunk_document(text)
            progress.update(task, description=f"Creating embeddings for {len(docs)} chunks...")
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            progress.update(task, description="✓ Document indexed successfully")
        console.print(f"[green]✓ Indexed {len(docs)} chunks from {file_path.name}[/green]")
        return len(docs)
    def ask_question(self, question: str) -> QAResult:
        """Ask a question about the indexed document"""
        if not self.vectorstore:
            raise ValueError("No document indexed. Please load a document first.")
        start_time = time.time()
        docs = self.vectorstore.similarity_search(question, k=self.top_k)
        context = "\n\n".join([f"[Chunk {d.metadata['chunk_id']}]: {d.page_content}"
                              for d in docs])
        qa_chain = self.qa_prompt | self.llm
        answer = qa_chain.invoke({"context": context, "question": question}).strip()
        latency = time.time() - start_time
        citations = [d.metadata.get('chunk_id', 'unknown') for d in docs]
        qa_result = QAResult(
            answer=answer,
            citations=citations,
            latency=latency
        )
        self._log_interaction(question, qa_result)
        return qa_result
    def _log_interaction(self, question: str, result: QAResult):
        """Log the QA interaction to JSONL file"""
        question_tokens = len(question.split())
        answer_tokens = len(result.answer.split())
        total_tokens = question_tokens + answer_tokens
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question": question,
            "answer": result.answer,
            "citations": result.citations,
            "latency": result.latency,
            "confidence": result.confidence,
            "token_usage": total_tokens,
            "model": {
                "embedding": self.embed_model,
                "llm": self.llm_model
            }
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    def display_result(self, question: str, result: QAResult):
        """Display QA result with citations"""
        answer_content = result.answer
        console.print(Panel(
            answer_content,
            title="[bold green]Answer[/bold green]",
            border_style="green"
        ))
        citations_text = ", ".join(result.citations)
        console.print(f"[dim]Citations: chunks {citations_text} | Latency: {result.latency:.2f}s[/dim]\n")

@app.command()
def main(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to text file"),
    ask: Optional[str] = typer.Option(None, "--ask", help="Ask a question directly"),
    chunk_size: int = typer.Option(None, "--chunk-size", help="Chunk size for text splitting"),
    chunk_overlap: int = typer.Option(None, "--chunk-overlap", help="Chunk overlap for text splitting")
):
    """Smart Document QA - Ask questions about your documents"""
    try:
        qa_system = SmartQA(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # Index document
        file_path = Path(input_path)
        qa_system.index_document(file_path)
        if ask:
            # Single question mode
            result = qa_system.ask_question(ask)
            qa_system.display_result(ask, result)
        else:
            # Interactive REPL mode
            console.print(Panel(
                "Interactive QA Mode\nType your questions or 'exit' to quit.",
                title="[bold cyan]Smart Document QA[/bold cyan]",
                border_style="cyan"
            ))
            while True:
                try:
                    question = input("Question >> ").strip()
                    if question.lower() in {"exit", "quit"}:
                        console.print("[cyan]Goodbye![/cyan]")
                        break
                    elif not question:
                        continue
                    result = qa_system.ask_question(question)
                    qa_system.display_result(question, result)
                except KeyboardInterrupt:
                    console.print("\n[cyan]Goodbye![/cyan]")
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
