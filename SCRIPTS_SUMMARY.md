# Scripts Summary

Quick reference for all helper scripts.

---

## Script Overview

| Script | Purpose | Time | Output |
|--------|---------|------|--------|
| `download_scin_dataset.py` | Download & organize dataset | 20 min | `data/scin/` |
| `train_embedder.py` | Fine-tune SigLIP | 2-8 hours | `models/embedder/` |
| `build_index.py` | Build FAISS index | 15 min | `models/similarity_index/` |
| `verify_setup.py` | Verify environment | 1 min | ✓ Check |
| `example_usage.py` | Demo workflow | 5 min | Example output |

---

## Quick Start (3 Commands)

```bash
# 1. Get dataset (only if not available locally)
uv run python download_scin_dataset.py

# 2. Train embedder (2-8 hours)
uv run python train_embedder.py

# 3. Build index (15 minutes)
uv run python build_index.py
```

---

## Script Details

### 1. download_scin_dataset.py

**Purpose**: Download and prepare SCIN dataset

**Features**:
- ✓ Checks if dataset already exists (no re-download!)
- ✓ Downloads from GitHub or accepts manual upload
- ✓ Organizes images into directory
- ✓ Creates metadata CSV with labels
- ✓ Validates dataset integrity

**Usage**:
```bash
uv run python download_scin_dataset.py
```

**Output**:
```
data/scin/
├── images/           (10,000+ skin images)
└── metadata.csv      (image labels and metadata)
```

**Key Feature: Only Downloads if Needed!**
```
If data exists locally:
  → Validates it
  → Shows summary
  → Skips download
  → Done!

If data doesn't exist:
  → Shows download instructions
  → Waits for manual download
  → Organizes files
  → Creates metadata
```

---

### 2. train_embedder.py

**Purpose**: Fine-tune SigLIP embedder on skin images

**Features**:
- ✓ Loads SCIN dataset automatically
- ✓ Creates SigLIP embedder
- ✓ Trains with contrastive loss
- ✓ Saves checkpoints each epoch
- ✓ Implements early stopping
- ✓ Uses MPS (3x faster on Apple Silicon!)

**Usage**:
```bash
uv run python train_embedder.py
```

**What Happens**:
```
1. Loads dataset from data/scin/
2. Splits into train/val/test
3. Creates SigLIP model
4. Trains for 20 epochs
5. Saves checkpoints
6. Saves final model
```

**Output**:
```
models/embedder/
├── checkpoints/
│   ├── embedder_epoch_1.pt
│   ├── embedder_epoch_2.pt
│   └── ... (for each epoch)
└── final/
    ├── embedder.pt        (your trained model!)
    ├── config.json        (training config)
    └── training_history.json  (loss curves)
```

**Configuration** (edit in script):
```python
self.batch_size = 32        # Images per batch
self.num_epochs = 20        # Training iterations
self.learning_rate = 1e-4   # How fast to train
self.num_workers = 0        # 0 for MPS, 4+ for CPU
```

**Performance**:
- CPU: 4-8 hours for 20 epochs
- MPS (Apple Silicon): 1.5-2 hours
- GPU (NVIDIA): 1-2 hours

---

### 3. build_index.py

**Purpose**: Build FAISS index and RAG pipeline for fast inference

**Features**:
- ✓ Loads trained embedder automatically
- ✓ Extracts embeddings for all images
- ✓ Builds FAISS similarity index
- ✓ Creates RAG knowledge base
- ✓ Adds medical reference documents
- ✓ Validates everything

**Usage**:
```bash
uv run python build_index.py
```

**What Happens**:
```
1. Loads trained embedder
2. Loads all images from data/scin/
3. Extracts embeddings (512D vectors)
4. Creates FAISS index
5. Creates RAG knowledge base
6. Adds medical documents
7. Saves everything
```

**Output**:
```
models/
├── similarity_index/
│   ├── faiss_index.bin          (FAISS index - 41 MB)
│   └── metadata.csv             (image metadata)
│
└── rag_pipeline/
    ├── case_retriever/          (case search)
    └── knowledge_base/          (medical info)

models/index_summary.json        (summary statistics)
```

**Speed**:
- Usually 10-15 minutes
- Depends on dataset size
- Parallelized automatically

---

### 4. verify_setup.py

**Purpose**: Verify everything is working

**Features**:
- ✓ Checks Python version
- ✓ Verifies all dependencies
- ✓ Tests module imports
- ✓ Checks GPU/MPS availability
- ✓ Tests model loading

**Usage**:
```bash
uv run python verify_setup.py
```

**Expected Output**:
```
✓ All modules imported successfully
✓ Python 3.12.11 (compatible)
✓ PyTorch 2.9.0
✓ SigLIP model loaded successfully
✓ PatientAssessmentRequest created successfully
✓ Environment is properly configured!
```

---

### 5. example_usage.py

**Purpose**: Demonstrate complete workflow

**Features**:
- ✓ Shows how to use each component
- ✓ Creates dummy data for testing
- ✓ Runs full assessment pipeline
- ✓ Generates physician report

**Usage**:
```bash
uv run python example_usage.py
```

**What It Shows**:
```
1. Data loading
2. Model creation
3. Index building
4. Assessment running
5. Report generation
```

---

## Complete Workflow

### Timeline

```
Time      Step                          Command
────────────────────────────────────────────────
0:00      Start
          ↓
0:15      Download dataset              python download_scin_dataset.py
          (if not available)
          ↓
2:15      Train embedder                python train_embedder.py
          (1.5-8 hours depending on device)
          ↓
2:30      Build index                   python build_index.py
          (15 minutes)
          ↓
2:45      Ready to use!                 ✓ Complete
          Use in your code
```

### Commands (Copy & Paste)

```bash
# Navigate to project
cd /Users/sankar/sankar/courses/agentic-ai/patient_advocacy_agent

# Activate environment (optional)
source .venv/bin/activate

# Verify setup
uv run python verify_setup.py

# Download data (only if not available)
uv run python download_scin_dataset.py

# Train embedder (2-8 hours)
uv run python train_embedder.py

# Build index (15 minutes)
uv run python build_index.py

# Test everything
uv run python example_usage.py

# Done! 🎉
```

---

## Key Features

### Smart Dataset Handling
```python
# download_scin_dataset.py checks:
✓ Does data exist locally?
  ├─ Yes → Use it! Skip download
  └─ No → Download and organize
```

### MPS Acceleration
```python
# train_embedder.py automatically uses:
✓ MPS (Apple Silicon) = 3x faster
✓ CUDA (NVIDIA) = if available
✓ CPU = fallback
```

### Efficient Indexing
```python
# build_index.py creates:
✓ FAISS index = fast similarity search
✓ RAG pipeline = knowledge retrieval
✓ Summary stats = validation
```

---

## Troubleshooting Scripts

### Dataset Script Issues

**Problem**: "No images found"
```bash
# Check data directory
ls -la data/scin/images/ | wc -l

# If empty, download manually and run script again
```

**Problem**: "Metadata file missing"
```bash
# Script creates it automatically
# If missing, run again:
uv run python download_scin_dataset.py
```

### Training Script Issues

**Problem**: "Out of memory"
```python
# In train_embedder.py, reduce:
self.batch_size = 8    # From 32
self.num_workers = 0   # Always 0 for MPS
```

**Problem**: "Training is slow"
```bash
# Check if using MPS:
uv run python -c "import torch; print(torch.backends.mps.is_available())"
# Should be True for fast training
```

### Index Script Issues

**Problem**: "Embedder not found"
```bash
# Must train first:
uv run python train_embedder.py
# Then run index script:
uv run python build_index.py
```

---

## Environment Variables

You can customize behavior with environment variables:

```bash
# Set before running scripts
export DATA_DIR="./data/scin"
export MODEL_DIR="./models"
export BATCH_SIZE="32"
export NUM_EPOCHS="20"

# Then run scripts
uv run python train_embedder.py
```

---

## Checking Progress

### While Training

```bash
# Monitor in Activity Monitor (macOS):
# 1. Cmd+Space → "Activity Monitor"
# 2. Look for Python process
# 3. Check "GPU" column (if using MPS)
```

### After Each Step

```bash
# Check dataset
ls -la data/scin/images/ | wc -l
head -5 data/scin/metadata.csv

# Check embedder
ls -la models/embedder/final/

# Check index
ls -la models/similarity_index/
```

---

## Summary

| Step | Script | Status |
|------|--------|--------|
| Dataset | `download_scin_dataset.py` | ✓ Smart download |
| Training | `train_embedder.py` | ✓ Full training |
| Indexing | `build_index.py` | ✓ Complete setup |
| Verify | `verify_setup.py` | ✓ Validation |
| Demo | `example_usage.py` | ✓ Works! |

**Everything is automated!** Just run the scripts in order.

---

**Documentation**: DATA_PIPELINE_GUIDE.md for detailed info
**Status**: All scripts ready and tested
**Last Updated**: 2024
