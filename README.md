# PyRagix

**Pronounced "Pie-Rajicks"** — A clean, typed, Pythonic pipeline for
Retrieval-Augmented Generation (RAG). Ingest HTML, PDF, and image-based
documents, build a FAISS vector store, and search with ease using Ollama for
answer generation. Designed for developers learning RAG, vector search, and
document processing in Python.

PyRagix is a lightweight, educational project to help you explore how to process
diverse documents (HTML, PDF, images) and enable intelligent search using modern
AI tools. It’s tuned for modest hardware (e.g., 16GB RAM / 6GB VRAM laptop) with
memory optimizations, but can be customized via `config.py`. It’s not a
production-grade unicorn—just a practical, well-structured example for
Pythonistas diving into RAG.

## Features

- **Document Ingestion**: Extract text from HTML, PDF, and images using
  `paddleocr` for OCR fallback, `pdfplumber`/`PyMuPDF` for PDFs, and
  BeautifulSoup for HTML.
- **Vector Store**: Build a FAISS index (FlatIP by default) with Sentence
  Transformers embeddings.
- **Console Search**: Query your document collection via an interactive
  command-line interface, with Ollama generating human-like answers from
  retrieved contexts.
- **Pythonic Design**: Clean, typed, idiomatic Python code with protocols,
  context managers, and memory cleanup for clarity and maintainability.
- **Memory Optimizations**: Tiled OCR for large pages, batch embedding, GPU/CPU
  thread capping, and automatic garbage collection.
- **Extensible**: Ready for future enhancements like a web interface or advanced
  FAISS indexing (e.g., IVF).

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/<your-username>/PyRagix.git
   cd PyRagix
   ```

2. **Set Up a Virtual Environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**: PyRagix uses a `requirements.in` file for
   dependency management. Ensure you have `pip` and `pip-tools` installed, then
   run:

   ```bash
   pip install pip-tools
   pip-compile requirements.in  # Generates requirements.txt
   pip install -r requirements.txt
   ```

   **Note**: The dependency list includes `torch`, `transformers`, `faiss-cpu`,
   `paddleocr`, `paddlepaddle`, `sentence-transformers`, `fitz` (PyMuPDF), and
   others. Ensure you have sufficient disk space and a compatible Python version
   (3.8+ recommended). For GPU acceleration, install CUDA-enabled versions where
   applicable.

4. **Ollama Setup** (for Querying):

   - Install Ollama: Follow instructions at [ollama.com](https://ollama.com).
   - Pull the default model: `ollama pull llama3.1:8b-instruct-q4_0`.
   - Start the Ollama server: `ollama serve`.

   Customize the Ollama model or URL in `query_rag.py` if needed.

## Usage

PyRagix consists of two main scripts:

- `ingest_folder.py`: Processes a folder of documents (HTML, PDF, images) and
  builds a FAISS vector store.
- `query_rag.py`: Runs an interactive console-based search interface to query
  the vector store and generate answers with Ollama.

### Step 1: Ingest Documents

Run the ingestion script to process a folder and create a FAISS index:

```bash
python ingest_folder.py [path/to/documents]
```

- If no folder is provided, it uses the default from `config.py` (e.g.,
  `./docs`).
- Supported formats: PDF, HTML/HTM, images (via OCR).
- Outputs: `local_faiss.index` (FAISS index), `documents.pkl` (metadata),
  `processed_files.txt` (processed file log), `ingestion.log` (processing log),
  and `crash_log.txt` (errors if any).
- Resumes from existing index if available; skips already processed files.

**Customization**: Edit `config.py` for hardware tuning (e.g., batch size,
thread counts, index type).

**Example**:

```bash
python ingest_folder.py ./my_documents
```

This scans `./my_documents` and subfolders, extracts text (with OCR fallback for
images/scans), chunks it, embeds with `all-MiniLM-L6-v2`, and adds to FAISS.

### Step 2: Search Documents

Launch the interactive console-based search interface:

```bash
python query_rag.py
```

- Loads the FAISS index and metadata.
- Enter queries at the prompt; get generated answers from Ollama based on
  retrieved contexts.
- Shows sources with scores and chunk indices.
- Type 'quit' or 'exit' to stop.

**Example Interaction**:

```
Query: What is machine learning?

Answer:
===========
Machine learning is a subset of AI that focuses on building systems that learn from data...
(Generated from Ollama using retrieved contexts)
===========

Sources:
1. intro.pdf (chunk 0, score: 0.920)
2. ml_basics.html (chunk 1, score: 0.850)
...
```

**Notes**:

- Ensure Ollama is running before starting `query_rag.py`.
- Customize Ollama settings (model, temperature, etc.) in `query_rag.py`'s
  `DEFAULT_CONFIG`.

## Configuration

- **config.py**: Tune for your hardware (e.g., thread limits, batch size, CUDA
  settings, FAISS index type).
- **query_rag.py**: Adjust Ollama API URL, model, or default top-K in
  `DEFAULT_CONFIG`.
- For larger setups: Increase batch sizes, use IVF indexing, or enable GPU
  FAISS.

## Requirements

PyRagix depends on a robust set of Python libraries for AI, document processing,
and vector search. Key dependencies include:

- `torch` and `transformers`/`sentence-transformers` for embedding models
- `faiss-cpu` for vector storage and search
- `paddleocr` and `paddlepaddle` for OCR
- `fitz` (PyMuPDF) and `pdfplumber` for PDF processing
- `beautifulsoup4` for HTML parsing
- `requests` for Ollama API calls

See [requirements.in](requirements.in) for the full list. Ensure your system has
sufficient resources for large models (e.g., `paddlepaddle` and `torch`).

## Contributing

We welcome contributions! If you’re learning RAG or want to enhance PyRagix,
here’s how to get started:

1. Fork the repo and create a feature branch.
2. Follow the installation steps above.
3. Submit a pull request with clear descriptions of your changes.

Ideas for contributions:

- Add support for more document formats (e.g., DOCX).
- Implement a web interface (planned for future releases).
- Optimize for different hardware (e.g., high-end GPUs or cloud).
- Enhance OCR handling or embedding models.

Please adhere to Python’s PEP 8 style guide and include type hints for
consistency.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.

## Acknowledgements

- Built with love for the Python and AI communities.
- Thanks to the creators of `faiss`, `sentence-transformers`, `paddleocr`,
  `ollama`, and `langchain` for their amazing tools.

Happy learning, and enjoy searching your documents with PyRagix! 🚀
