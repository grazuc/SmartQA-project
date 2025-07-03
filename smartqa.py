import os
from pathlib import Path
from dotenv import load_dotenv
import typer
from rich.console import Console
from typing import Optional, List

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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
        self.processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)
        self.vectorstore = None

    def index_document(self, file_path: Path) -> int:
        text = self.processor.load_document(file_path)
        docs = self.processor.chunk_document(text)
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        console.print(f"[green]✓ Indexed {len(docs)} chunks from {file_path.name}[/green]")
        return len(docs)

@app.command()
def chunk(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to text file"),
    chunk_size: int = typer.Option(None, "--chunk-size", help="Chunk size for text splitting"),
    chunk_overlap: int = typer.Option(None, "--chunk-overlap", help="Chunk overlap for text splitting")
):
    load_env()
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    file_path = Path(input_path)
    text = processor.load_document(file_path)
    chunks = processor.chunk_document(text)
    console.print(f"[green]✓ Loaded {file_path.name} and generated {len(chunks)} chunks.[/green]")
    for i, chunk in enumerate(chunks):
        preview = chunk.page_content[:60].replace('\n', ' ')
        console.print(f"[cyan]Chunk {i}:[/cyan] {preview}...")


@app.command()
def index(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to text file"),
    chunk_size: int = typer.Option(None, "--chunk-size", help="Chunk size for text splitting"),
    chunk_overlap: int = typer.Option(None, "--chunk-overlap", help="Chunk overlap for text splitting")
):
    """
    Index a document (chunks + embeddings) and report the number of chunks.
    """
    qa = SmartQA(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    file_path = Path(input_path)
    n_chunks = qa.index_document(file_path)
    console.print(f"[green]Document indexed with {n_chunks} chunks.[/green]")

@app.command()
def main():
    load_env()
    print("SmartQA bootstrap loaded.")

if __name__ == "__main__":
    app()
