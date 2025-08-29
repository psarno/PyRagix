# Claude Code Development Rules

## Core Python Standards

# ALWAYS follow PEP 8 and use descriptive snake_case names

# ALWAYS prefix private functions with underscore (\_private_function)

# ALWAYS include type hints and docstrings for functions

# ALWAYS use pathlib.Path for file operations

# ALWAYS use context managers (with statements) for resources

# NEVER use bare except clauses - specify exception types

# NEVER hardcode file paths - use configuration

## RAG Pipeline Essentials

# ALWAYS validate inputs for both ingest_folder.py and query_rag.py

# ALWAYS implement proper error handling for FAISS operations

# ALWAYS use logging module instead of print() for non-user output

# ALWAYS handle memory cleanup with context managers for large operations

# NEVER ignore exceptions during document processing - log and continue

# NEVER assume vector store exists - check before loading

## Code Quality

# ALWAYS check Pylance static analysis after major changes or TODO implementations

# ALWAYS fix Pylance errors and warnings before considering code complete

# ALWAYS write clear docstrings explaining function purpose, params, and returns

# NEVER leave unresolved Pylance static analysis issues

## CLI Best Practices

# ALWAYS provide helpful command-line help and error messages

# ALWAYS handle KeyboardInterrupt gracefully

# ALWAYS return appropriate exit codes (0 success, non-zero errors)

## Dependencies & Versions

# ALWAYS check requirements.in first for package versions, then requirements.txt if needed for full dependency details

# NEVER suggest upgrading existing packages - current versions are tested and working

# PyTorch is CUDA-enabled (torch==2.8.0+cu126) - maintain CUDA compatibility

# If adding NEW dependencies: add to requirements.in then run pip-compile to update requirements.txt

# NEVER suggest pip install commands without checking current versions first

# NEVER add lines such as "Generated with Claude Code" to any project files.

# ALWAYS format release notes and documentation without artificial line breaks - use full sentences on single lines for easy copy-paste

# ALWAYS link release note items to specific commit hashes when available - format as "description (commit_hash)"

---

**Post-major-changes checklist**: Check VS Code Problems panel for Pylance
issues, test edge cases, verify error handling, update docstrings
