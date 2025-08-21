@echo off

REM --- Activate the Python virtual environment ---
echo Actvating virtal environment ...
call rag-env\scripts\activate

REM --- Add PyTorch lib to PATH for DLL loading ---
set "PATH=G:\RAG-Project\rag-env\lib\site-packages\torch\lib;%PATH%"

REM --- Launch RAG ingestion with safe memory settings ---

REM --- Limit NumPy / BLAS threading ---
REM --- Doesn’t change global Windows settings — only applies for that run. ---
REM --- Adjust the =2 to =4 if you want more CPU parallelism (more speed, more RAM). ---
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1

set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512      
set CUDA_VISIBLE_DEVICES=0

REM Run Python
python ingest_folder.py

pause
