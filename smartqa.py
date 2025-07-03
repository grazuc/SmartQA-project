import os
from dotenv import load_dotenv
import typer

app = typer.Typer(help="Smart Document QA - Ask questions about your .txt documents")

def load_env():
    load_dotenv()

@app.command()
def main():
    """Entry point CLI."""
    load_env()
    print("SmartQA bootstrap loaded.")

if __name__ == "__main__":
    app()
