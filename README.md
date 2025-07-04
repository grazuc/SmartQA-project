# Smart Document QA

A lightweight CLI tool that ingests plain-text documents and lets you ask questions about their contents using LLM-powered retrieval-augmented generation.

## Features

- 📄 **Document Ingestion**: Load and chunk `.txt` files for efficient querying
- 🔍 **Semantic Search**: Uses embeddings to find relevant content chunks
- 💬 **Interactive Mode**: REPL interface for multiple questions
- 📊 **Usage Logging**: Tracks questions, answers, and performance metrics
- 🎯 **Confidence Scoring**: Optional confidence assessment for answers
- 🎨 **Rich Output**: Colorized terminal output with citations

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Tests

```bash
pytest
```
Tests cover document processing, chunking, retrieval, and core QA functionality.

### Usage

**Single Question Mode:**
```bash
python smartqa.py --input API.txt --ask "What is the main purpose of this API?"
```

**Interactive Mode:**
```bash
python smartqa.py --input API.txt
```

**With Confidence Scoring:**
```bash
python smartqa.py --input API.txt --ask "Your question here" --score
```

## Configuration

Create a `.env` file to customize settings (you can use the .env.example uploaded too):

```env
# Model Configuration
EMBED_MODEL=intfloat/multilingual-e5-base
LLM_MODEL=google/flan-t5-small

# Chunking Parameters
CHUNK_SIZE=300
CHUNK_OVERLAP=40
TOP_K=3

# Logging
LOG_FILE=qa_history.jsonl
```

## Command Line Options

- `--input, -i`: Path to input text file (required)
- `--ask`: Ask a single question and exit
- `--score`: Include confidence scoring (0-1 scale)
- `--chunk-size`: Override default chunk size
- `--chunk-overlap`: Override default chunk overlap

## Example Files

- `API.txt` - Sample document for testing the tool

## Output

The tool provides:
- **Answer**: Generated response (max 200 words)
- **Citations**: Referenced chunk IDs
- **Latency**: Response time
- **Confidence**: Optional certainty score (with `--score` flag)

All interactions are logged to `qa_history.jsonl` for analysis.

## Architecture

- **Document Processing**: Recursive text splitting with configurable chunk sizes
- **Embeddings**: HuggingFace multilingual embeddings for semantic search
- **Vector Store**: FAISS for efficient similarity search
- **LLM**: HuggingFace text generation models
- **Logging**: JSONL format for structured interaction history

## Requirements

- Python ≥ 3.9
- See `requirements.txt` for dependencies
