import os
from pathlib import Path
from dotenv import load_dotenv
import typer
from rich.console import Console
from typing import Optional, List

# CLI setup
app = typer.Typer(help="Smart Document QA - Ask questions about your .txt documents")
console = Console()

def load_env():
    load_dotenv()


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentProcessor:
    """
    Handles document loading and chunking. You can change the chunk_size and overlap by CLI or .env.
    """
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
        """Split document into chunks using chunk_size and chunk_overlap"""
        docs = self.splitter.create_documents([text])
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "chunk_id": str(i),
                "source": "input_document"
            })
        return docs

# --- Comando para probar chunking ---
@app.command()
def chunk(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to text file"),
    chunk_size: int = typer.Option(None, "--chunk-size", help="Chunk size for text splitting"),
    chunk_overlap: int = typer.Option(None, "--chunk-overlap", help="Chunk overlap for text splitting")
):
    """
    Test document loading and chunking with optional parameters.
    """
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
def main():
    """Entry point CLI."""
    load_env()
    print("SmartQA bootstrap loaded.")

if __name__ == "__main__":
    app()
