# Doc Sage

A document Q&A assistant that uses RAG (Retrieval-Augmented Generation) to answer questions about PDF documents with page-level citations.

## Features

- PDF processing with page-level chunking
- Semantic search using cosine similarity (threshold: 0.3)
- Streaming responses with conversation history
- Precise page citations for transparency

## Requirements

- Python 3.13+
- LM Studio running on `http://localhost:1234`
- Models: `text-embedding-nomic-embed-text-v1.5`, `qwen/qwen3-4b`

## Quick Start

```bash
# Install dependencies
uv sync

# Or with pip
pip install lmstudio scikit-learn pymupdf
```

```python
from main import RAG

rag = RAG("document.pdf")
rag.chat("What are the main findings?")  # Streams response to console
```

## How it Works

1. **Extract** - Splits PDF into page-based chunks with metadata
2. **Embed** - Creates embeddings for query and document pages  
3. **Retrieve** - Finds top 7 most similar chunks (cosine similarity ≥ 0.3)
4. **Generate** - Uses LM Studio to create contextual responses with page citations
