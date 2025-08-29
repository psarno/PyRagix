# Claude Code Development Rules

## Core Python Standards

# ALWAYS prefix private functions with underscore (\_private_function)

# NEVER hardcode file paths - use configuration

## RAG Pipeline Essentials

# ALWAYS validate inputs for both ingest_folder.py and query_rag.py

# ALWAYS implement proper error handling for FAISS operations

# ALWAYS handle memory cleanup with context managers for large operations

# NEVER ignore exceptions during document processing - log and continue

# NEVER assume vector store exists - check before loading

## Code Quality

# ALWAYS check Pylance static analysis after major changes or TODO implementations

# ALWAYS fix Pylance errors and warnings before considering code complete

# NEVER leave unresolved Pylance static analysis issues

## CLI Best Practices

## Dependencies & Versions

# ALWAYS check requirements.in first for package versions, then requirements.txt if needed for full dependency details

# NEVER suggest upgrading existing packages - current versions are tested and working

# PyTorch is CUDA-enabled (torch==2.8.0+cu126) - maintain CUDA compatibility

# If adding NEW dependencies: add to requirements.in then run pip-compile to update requirements.txt

# NEVER suggest pip install commands without checking current versions first

# NEVER add lines such as "Generated with Claude Code" to any project files.

# ALWAYS format release notes and documentation without artificial line breaks - use full sentences on single lines for easy copy-paste

# ALWAYS link release note items to specific commit hashes when available - format as "description (commit_hash)"

# ALWAYS Ceate and save the README.md file directly to the filesystem with no line length restrictions, rather than showing it in the chat window.

# ALWAYS Format all markdown files with no automatic line breaks - only break lines for actual paragraphs, lists, or code blocks. Do not wrap text at 80 characters.

---

**Post-major-changes checklist**: Check VS Code Problems panel for Pylance
issues, test edge cases, verify error handling, update docstrings
