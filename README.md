# Simple RAG App

A document analysis assistant that uses RAG (Retrieval-Augmented Generation) to answer questions about uploaded PDF documents.

## Features

- **Page-level chunking** - Splits PDFs into page-based chunks for precise citations
- **Semantic search** - Uses embeddings and cosine similarity to find relevant content
- **Document-grounded responses** - Answers questions based primarily on uploaded content
- **Page citations** - References specific pages when providing answers

## Requirements

- Python 3.13+
- LM Studio running locally on `http://localhost:1234`
- Models: `text-embedding-nomic-embed-text-v1.5` (embeddings), `qwen/qwen3-4b` (LLM)

## Installation

```bash
uv add lmstudio scikit-learn pymupdf
```

## Usage

```python
from main import RAG

# Initialize with PDF path
rag = RAG("document.pdf")

# Ask questions about the document
response = rag.generate_summary("What are the main findings?")
print(response)
```

## How it works

1. **Document Processing** - Extracts text from each PDF page with metadata
2. **Retrieval** - Embeds query and document pages, finds most similar content
3. **Generation** - Uses retrieved content to generate contextual responses
4. **Citation** - Maintains page-level references for transparency

## Architecture

- `RAG` class handles the complete pipeline
- Page-level chunking preserves document structure
- Cosine similarity for semantic retrieval
- LM Studio integration for local LLM inference
