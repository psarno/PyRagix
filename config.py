# ======================================
# Config for ingestion script
# Default: tuned for 16GB RAM / 6GB VRAM laptop
# ======================================

# CPU / Threading
TORCH_NUM_THREADS = 2
OPENBLAS_NUM_THREADS = 2
MKL_NUM_THREADS = 2
OMP_NUM_THREADS = 2
NUMEXPR_MAX_THREADS = 2

# GPU / CUDA
CUDA_VISIBLE_DEVICES = "0"  # single GPU
PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:1024,garbage_collection_threshold:0.9"

# Sentence-Transformers
BATCH_SIZE = 16  # safe for 6GB GTX 1660 Ti

# FAISS

# Under ~50k–100k chunks
INDEX_TYPE = "flat"  # "ivf_flat", "ivf_pq", or "flat"
NLIST = 1024
NPROBE = 16

# If currently beyond 100k chunks, use:
# INDEX_TYPE = "ivf_flat"
# NLIST = 1024
# NPROBE = 16

# Files to skip during processing (by filename)
SKIP_FILES = {
    "UNC - International Biosafety Committee Meeting Minutes 2019 - Redacted.pdf"
}
