# PyRagix

A clean, typed, Pythonic pipeline for Retrieval-Augmented Generation (RAG).
Ingest HTML, PDF, and image-based documents, build a FAISS vector store, and
search with ease using Ollama for answer generation. Designed for developers
learning RAG, vector search, and document processing in Python.

PyRagix is a lightweight, educational project to help you explore how to process
diverse documents (HTML, PDF, images) and enable intelligent search using modern
AI tools. It's tuned for modest hardware (e.g., 16GB RAM / 6GB VRAM) with memory
optimizations, but can be customized via `settings.json`. This project is meant
to be a practical, well-structured example for Python developers diving into
RAG.

## Features

- **Cross-Platform**: Runs natively on Windows, Linux, and macOS with identical
  functionality. Uses `pathlib` for universal file handling.
- **Document Ingestion**: Extract text from HTML, PDF, and images using
  `PaddleOCR` for OCR fallback, `PyMuPDF` for PDFs, and BeautifulSoup for HTML.
- **Vector Store**: Build a FAISS index with Sentence Transformers embeddings.
  Supports both Flat and IVF (Inverted File) indexing for optimal performance
  scaling.
- **Console Search**: Query your document collection via an interactive
  command-line interface, with Ollama generating human-like answers from
  retrieved contexts.
- **Web Interface**: Modern, responsive web UI for searching documents with
  real-time status indicators, configurable options, and beautiful results
  presentation.
- **Pythonic Design**: Clean, typed, idiomatic Python code with protocols,
  context managers, and memory cleanup for clarity and maintainability.
- **Memory Optimizations**: Adaptive memory settings based on system RAM, tiled
  OCR for large pages, batch embedding with retry logic, and automatic garbage
  collection.
- **Modular Architecture**: Separate classes for OCR processing and
  configuration management for better code organization and testing.
- **Advanced Indexing**: Configurable FAISS indexing with IVF support for faster
  search on large datasets, with intelligent fallback for robust operation.
- **Hybrid CPU/GPU Support**: Automatic detection of GPU FAISS capabilities with
  graceful fallback to CPU-only operation for universal compatibility.
- **Modern Web Interface**: Complete TypeScript/FastAPI web application with
  professional dark theme, real-time search, and responsive design.

## Project Structure

```
PyRagix/
├── ingest_folder.py        # Main ingestion script
├── query_rag.py           # RAG query interface (console)
├── web_server.py          # FastAPI web server
├── start_web.bat          # Web interface startup script (Windows)
├── start_web.sh           # Web interface startup script (Linux/Mac)
├── ingest.bat             # Document ingestion script (Windows)
├── ingest.sh              # Document ingestion script (Linux/Mac)
├── query.bat              # Query interface script (Windows)
├── query.sh               # Query interface script (Linux/Mac)
├── config.py              # Configuration loader and validation
├── settings.json          # User configuration file (auto-generated)
├── classes/
│   ├── ProcessingConfig.py # Data class for processing configuration
│   └── OCRProcessor.py     # OCR operations handler
├── web/                   # Web interface files
│   ├── index.html         # Main web interface
│   ├── style.css          # Modern dark theme styling
│   ├── script.ts          # TypeScript source (ES2024)
│   ├── script.js          # Compiled JavaScript
│   ├── tsconfig.json      # TypeScript configuration
│   └── dev.bat           # TypeScript development script
├── requirements.in         # Package dependencies (source)
├── requirements.txt        # Compiled dependencies
├── local_faiss.index      # Generated FAISS vector index
├── documents.db           # Document metadata database
├── processed_files.txt    # Log of processed files
├── ingestion.log         # Processing logs
└── crash_log.txt         # Error logs (when failures occur)
```

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/<your-username>/PyRagix.git
   cd PyRagix
   ```

2. **Set Up a Virtual Environment** (recommended):

   ```bash
   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   
   # Windows
   python -m venv venv
   venv\Scripts\activate.bat
   ```

3. **Install Dependencies**: 

   ```bash
   # Most users - install from pinned versions
   pip install -r requirements.txt
   ```

   **For developers modifying dependencies**: PyRagix uses `requirements.in` for dependency management. To update dependencies:

   ```bash
   pip install pip-tools           # Required for pip-compile command
   pip-compile requirements.in     # Updates requirements.txt
   pip install -r requirements.txt
   ```

   **Note**: The dependency list includes `torch`, `transformers`, `faiss-cpu`,
   `paddleocr`, `paddlepaddle`, `sentence-transformers`, `fitz` (PyMuPDF),
   `fastapi`, `uvicorn`, and others. Ensure you have sufficient disk space and a
   compatible Python version (3.8+ recommended). For GPU acceleration, install
   CUDA-enabled versions where applicable.

4. **Ollama Setup** (for Querying):

   - Install Ollama: Follow instructions at [ollama.com](https://ollama.com).
   - Pull the default model: `ollama pull llama3.2:3b-instruct-q4_0`.
   - Start the Ollama server: `ollama serve`.

   Customize the Ollama model or URL in `query_rag.py` if needed.

## Usage

PyRagix provides both console and web interfaces for document search:

- `ingest_folder.py`: Processes a folder of documents (HTML, PDF, images) and
  builds a FAISS vector store.
- `query_rag.py`: Interactive console-based search interface.
- `web_server.py`: Modern web interface with REST API backend.

### Step 1: Ingest Documents

Run the ingestion script to process a folder and create a FAISS index:

```bash
# Direct Python command (all platforms)
python ingest_folder.py [options] [path/to/documents]

# Or use convenience scripts:
# Windows
ingest.bat [options] [path/to/documents]

# Linux/Mac
./ingest.sh [options] [path/to/documents]
```

**Command Options:**
- `--fresh`: Start from scratch, clearing existing index and processed files log
- `--no-recurse`: Only process files in the root folder, skip subdirectories

- If no folder is provided, it uses the default from `config.py` (e.g.,
  `./docs`).
- Supported formats: PDF, HTML/HTM, images (via OCR).
- Outputs: `local_faiss.index` (FAISS index), `documents.db` (metadata database),
  `processed_files.txt` (processed file log), `ingestion.log` (processing log),
  and `crash_log.txt` (errors if any).
- Resumes from existing index if available; skips already processed files.

**Customization**: Edit `settings.json` for hardware tuning (e.g., batch size,
thread counts, index type). The file is auto-generated on first run with optimal
defaults for your system. IVF indexing is enabled by default for better
performance scaling.

**Examples**:

```bash
# Basic usage - process folder and subdirectories (default)
python ingest_folder.py ./my_documents

# Fresh start - clear existing index and start over
python ingest_folder.py --fresh ./my_documents

# Process only root folder - skip subdirectories
python ingest_folder.py --no-recurse ./my_documents

# Fresh start with no subdirectories
python ingest_folder.py --fresh --no-recurse ./my_documents
```

These commands scan the specified folder, extract text (with OCR fallback for images/scans), chunk it, embed with `all-MiniLM-L6-v2`, and add to a FAISS IVF index optimized for fast retrieval.

### Step 2: Search Documents

PyRagix offers two search interfaces:

#### Option A: Web Interface (Recommended)

Launch the modern web interface:

```bash
# Windows (using convenience script)
start_web.bat

# Linux/Mac (using convenience script)
./start_web.sh

# Direct Python command (all platforms)
python web_server.py
```

Then open your browser to:
- **Web Interface**: http://localhost:8000/web/
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

**Web Interface Features:**
- Modern, responsive dark theme design
- Real-time server status indicator
- Configurable search options (results count, sources, debug mode)
- Beautiful answer presentation with source highlighting
- TypeScript-powered frontend with ES2024 features
- REST API backend for integration

#### Option B: Console Interface

Launch the interactive console-based search interface:

```bash
# Direct Python command (all platforms)
python query_rag.py

# Or use convenience scripts:
# Windows
query.bat

# Linux/Mac
./query.sh
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

**Platform Notes**:

- **All platforms**: Core Python functionality is identical across Windows, Linux, and macOS
- **Convenience scripts**: Both `.bat` (Windows) and `.sh` (Linux/Mac) scripts provided for all operations
- **Windows users**: Use `start_web.bat`, `ingest.bat`, `query.bat` 
- **Linux/Mac users**: Use `./start_web.sh`, `./ingest.sh`, `./query.sh` (scripts are executable)
- **TypeScript development**: Requires `npm install -g typescript` for compilation
- Ensure Ollama is running before starting queries on any platform

## Configuration

- **settings.json**: Main configuration file for hardware tuning (e.g., thread
  limits, batch size, CUDA settings, FAISS index type). Auto-generated with
  system-appropriate defaults.
- **classes/ProcessingConfig.py**: Adaptive configuration that automatically
  adjusts memory settings based on available system RAM.
- **query_rag.py**: Ollama API settings loaded from `settings.json` via
  `config.py`.

### FAISS Index Types

PyRagix supports two FAISS index types via the `INDEX_TYPE` setting:

- **"ivf"** (default): IVF (Inverted File) indexing for faster searches on large
  datasets. Automatically falls back to flat indexing for smaller collections
  (< ~2048 chunks), then upgrades to IVF as your document collection grows.
  Configurable via `NLIST` (clusters, default: 1024) and `NPROBE` (search clusters, default: 16).
- **"flat"**: Flat indexing for exhaustive search. Slower but more accurate.
  Use this if you want to force flat indexing regardless of collection size.

Optimal settings for modest hardware (16GB RAM, 6GB VRAM):

```json
{
  "INDEX_TYPE": "ivf",
  "NLIST": 1024,
  "NPROBE": 16
}
```

### GPU Acceleration

PyRagix includes intelligent GPU detection and hybrid CPU/GPU support:

- **Automatic Detection**: Detects if GPU FAISS functions are available
- **Graceful Fallback**: Uses CPU when GPU unavailable (default behavior)
- **Configurable**: Enable GPU acceleration via `settings.json`:

```json
{
  "GPU_ENABLED": true,
  "GPU_DEVICE": 0,
  "GPU_MEMORY_FRACTION": 0.8
}
```

**Note**: GPU FAISS requires compatible hardware and special installation. The system works perfectly with CPU-only FAISS (default) and will automatically utilize GPU capabilities when available.

For larger setups: Increase `NLIST` (more clusters) and `NPROBE` values.

## Advanced Configuration

PyRagix provides extensive configuration options in `settings.json` for fine-tuning performance and behavior. Here's a breakdown of the more technical parameters:

### Performance & Threading

- **`TORCH_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `NUMEXPR_MAX_THREADS`**: Control CPU parallelism for different math libraries. Default is 6 threads. Increase for high-core CPUs, decrease for shared systems or to reduce memory usage.

- **`BATCH_SIZE`**: Number of documents processed simultaneously during embedding (default: 16). Larger values use more memory but can be faster. Reduce if you encounter out-of-memory errors.

- **`BATCH_SIZE_RETRY_DIVISOR`**: When batch processing fails due to memory, the batch size is divided by this value (default: 4) and retried. Higher values mean more aggressive fallback.

### CUDA Memory Management

- **`PYTORCH_CUDA_ALLOC_CONF`**: Advanced CUDA memory allocation settings:
  - `max_split_size_mb:1024`: Maximum size (MB) for memory block splitting. Larger values reduce fragmentation but use more memory.
  - `garbage_collection_threshold:0.9`: Triggers cleanup when 90% of allocated memory is used. Lower values free memory more aggressively.

### OCR Processing

- **`BASE_DPI`**: Resolution for OCR processing (default: 150). Higher values (200-300) improve text recognition accuracy but increase processing time and memory usage. Lower values (100-120) speed up processing for simple documents.

### Document Processing

- **`SKIP_FILES`**: Array of file patterns to ignore during ingestion (e.g., `["*.tmp", "backup_*"]`). Supports glob patterns.

- **`INGESTION_LOG_FILE`, `CRASH_LOG_FILE`**: Customize log file names for processing events and errors.

### LLM Generation Parameters

- **`TEMPERATURE`**: Controls response creativity (0.0-1.0, default: 0.1). Lower values produce more focused, deterministic answers. Higher values increase creativity but may reduce accuracy.

- **`TOP_P`**: Nucleus sampling parameter (default: 0.9). Controls diversity by only considering tokens comprising the top 90% probability mass. Lower values make responses more focused.

- **`MAX_TOKENS`**: Maximum length of generated answers (default: 500). Increase for longer responses, decrease to save time and tokens.

- **`DEFAULT_TOP_K`**: Number of document chunks retrieved for each query (default: 7). More chunks provide richer context but may include less relevant information.

- **`REQUEST_TIMEOUT`**: Ollama API timeout in seconds (default: 60). Increase for complex queries or slower models.

### Tuning Tips

- **Memory-constrained systems**: Reduce `BATCH_SIZE` to 8 or lower, decrease thread counts to 2-4, and set `BASE_DPI` to 100.
- **High-performance systems**: Increase thread counts to match CPU cores, raise `BATCH_SIZE` to 32+, and use `BASE_DPI` 200-300 for better OCR.
- **Better answers**: Increase `DEFAULT_TOP_K` to 10-15, raise `MAX_TOKENS` to 800-1000, and fine-tune `TEMPERATURE` (0.2-0.3 for creative but focused responses).

## Requirements

PyRagix depends on a robust set of Python libraries for AI, document processing,
and vector search. Key dependencies include:

- `torch` and `transformers`/`sentence-transformers` for embedding models
- `faiss-cpu` for vector storage and search (with optional GPU support detection)
- `paddleocr` and `paddlepaddle` for OCR operations
- `fitz` (PyMuPDF) for PDF processing
- `beautifulsoup4` (with optional `lxml`) for HTML parsing
- `requests` for Ollama API calls
- `fastapi` and `uvicorn` for the web interface and REST API
- `sqlite-utils` for metadata database operations
- `psutil` for system memory detection

See [requirements.in](requirements.in) for the complete dependency list and
`requirements.txt` for pinned versions. The system automatically adapts memory
settings based on available RAM (16GB+ recommended for optimal performance).

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
