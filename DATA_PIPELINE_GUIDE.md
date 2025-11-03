# Data Pipeline Guide

Complete guide to downloading the SCIN dataset and building the patient advocacy agent.

---

## Overview

There are three main scripts to get your system ready:

1. **download_scin_dataset.py** - Download and organize SCIN dataset
2. **train_embedder.py** - Fine-tune SigLIP embedder
3. **build_index.py** - Build FAISS index and RAG pipeline

---

## Step 1: Download SCIN Dataset

### About SCIN

**SCIN** (Skin Condition Image Network) is a public dataset of skin condition images with labels.

- **Size**: 10,000+ images
- **Conditions**: Eczema, Psoriasis, Acne, Dermatitis, etc.
- **Format**: JPEG/PNG images + metadata CSV
- **License**: Check repository for license terms
- **Source**: https://github.com/ISMAE-SUDA/SCIN

### Run Download Script

```bash
cd /Users/sankar/sankar/courses/agentic-ai/patient_advocacy_agent
uv run python download_scin_dataset.py
```

### What the Script Does

```
1. Checks if dataset already exists
2. Creates directories (data/scin/)
3. Shows download instructions
4. Helps organize downloaded files
5. Creates metadata CSV
6. Validates dataset
```

### Expected Output

```
================================================================================
SCIN Dataset Download and Preparation Tool
================================================================================

Step 1: Download Dataset
================================================================================
[Download instructions...]

Step 2: Organize Dataset
================================================================================
✓ Found 10234 images

Step 3: Create Metadata
================================================================================
✓ Created metadata file: data/scin/metadata.csv
  - Total images: 10234
  - Conditions: 8

Step 4: Validate Dataset
================================================================================
✓ Dataset validation passed!

✓ Dataset is ready to use!
```

### Manual Download (Alternative)

If automated download doesn't work:

1. Visit: https://github.com/ISMAE-SUDA/SCIN
2. Download the repository (ZIP)
3. Extract to `data/scin/`
4. Run the script again

### Alternative Datasets

You can use these instead of SCIN:

**A. ISIC (Skin Lesion Analysis)**
- URL: https://www.isic-archive.com/
- Size: 25,000+ images
- Format: Multiple formats available
- Usage: Download and extract to `data/isic/`

**B. DermNet**
- URL: https://www.dermnetnz.org/
- Format: Various, see documentation
- Usage: Requires manual organization

**C. Fitzpatrick 17k**
- URL: https://github.com/mattgroff/fitzpatrick17k
- Size: 16,977 images
- Format: Well-organized, easy to use

**D. Create Your Own**
- Collect images for your specific use case
- Organize into folders by condition
- Create metadata CSV manually

---

## Step 2: Fine-tune Embedder

### About This Step

This step trains the SigLIP embedder on your dataset using contrastive loss:

- **Input**: Raw skin condition images
- **Output**: Fine-tuned embedding model
- **Goal**: Learn visual similarity between skin conditions
- **Time**: 1-4 hours on MPS, 4-8 hours on CPU

### Run Training Script

```bash
uv run python train_embedder.py
```

### What the Script Does

```
1. Loads SCIN dataset
2. Creates SigLIP embedder
3. Sets up trainer with contrastive loss
4. Trains for 20 epochs (configurable)
5. Saves checkpoints after each epoch
6. Performs early stopping if needed
7. Saves final model and training history
```

### Expected Output

```
================================================================================
Training Configuration
================================================================================
Device: mps
Data dir: ./data/scin
Batch size: 32
Epochs: 20
Model: google/siglip-base-patch16-224

================================================================================
Loading Dataset
================================================================================
✓ Dataset loaded
  - Train: 7164 images
  - Val: 1023 images
  - Test: 2047 images
  - Conditions: 8

================================================================================
Creating Embedder
================================================================================
✓ Embedder created
  - Model: google/siglip-base-patch16-224
  - Projection dim: 512

================================================================================
Training Embedder
================================================================================
Starting training...
Epoch 1/20 - Train Loss: 2.1234, Val Loss: 1.9876
Epoch 2/20 - Train Loss: 1.8765, Val Loss: 1.7654
...
Epoch 20/20 - Train Loss: 0.5432, Val Loss: 0.6789

✓ Training completed
  - Final train loss: 0.5432
  - Final val loss: 0.6789

✓ Model saved to ./models/embedder/final/embedder.pt
✓ Training history saved
✓ Config saved
```

### Training Configuration

Edit the values in the script to customize:

```python
class TrainingConfig:
    self.batch_size = 32           # Adjust for your GPU memory
    self.num_epochs = 20           # Number of training epochs
    self.learning_rate = 1e-4      # Learning rate
    self.num_workers = 0           # 0 for MPS, 4+ for CPU
```

### Troubleshooting Training

**Issue: Out of Memory**
```python
# Reduce batch size
self.batch_size = 16  # or 8
self.num_workers = 0  # Always 0 for MPS
```

**Issue: Training is Slow**
```bash
# Check device being used
uv run python -c "import torch; print('MPS:', torch.backends.mps.is_available())"

# Should show MPS: True for faster training
```

**Issue: Training crashes**
```python
# Use smaller dataset for testing
self.batch_size = 8
self.num_epochs = 2

# Then increase if it works
```

---

## Step 3: Build FAISS Index and RAG

### About This Step

This step creates a searchable index for fast similarity matching:

- **Input**: Fine-tuned embedder + dataset
- **Output**: FAISS index + RAG pipeline
- **Goal**: Enable instant retrieval of similar cases
- **Time**: 5-15 minutes depending on dataset size

### Run Index Building Script

```bash
uv run python build_index.py
```

### What the Script Does

```
1. Loads trained embedder
2. Loads metadata CSV
3. Extracts embeddings for all images
4. Builds FAISS similarity index
5. Creates RAG knowledge base
6. Adds medical reference documents
7. Saves everything
```

### Expected Output

```
================================================================================
Build FAISS Index and RAG Pipeline
================================================================================

Configuration:
  Data dir: ./data/scin
  Embedder: ./models/embedder/final/embedder.pt
  Index dir: ./models/similarity_index
  Device: mps

================================================================================
Loading Embedder
================================================================================
✓ Embedder loaded from ./models/embedder/final/embedder.pt
✓ Metadata loaded (10234 images)

================================================================================
Extracting Embeddings
================================================================================
Extracting embeddings (batch_size=32)...
  train: 100 images processed
  train: 200 images processed
  ...
✓ Extracted embeddings for 10234 images
  Shape: (10234, 512)

================================================================================
Building Similarity Index
================================================================================
✓ Index created
✓ Index saved to ./models/similarity_index

================================================================================
Building RAG Pipeline
================================================================================
Creating case retriever...
✓ Case retriever created
Creating knowledge base...
✓ Knowledge base created with 5 documents
✓ RAG pipeline created
✓ RAG pipeline saved to ./models/rag_pipeline

✓ Index summary saved to ./models/index_summary.json

Index Statistics:
  - Total images: 10234
  - Conditions: 8
  - Embeddings size: 41.2 MB

================================================================================
Index Building Complete!
================================================================================
```

### Using the Built Index

Once the index is built, you can use it for assessments:

```python
from patient_advocacy_agent import PatientAssessmentAPI

api = PatientAssessmentAPI(agent, embedder)
assessment = api.assess_patient(request)
```

---

## Complete Workflow

### Timeline Example

```
Time    Step                    Duration    Device
──────────────────────────────────────────────────
0:00    Start
0:15    Download dataset        15 min      Network
0:15    Organize files          5 min       CPU
0:20    Train embedder          2 hours     MPS
2:20    Extract embeddings      10 min      MPS
2:30    Build index             5 min       CPU
2:35    COMPLETE!               ✓
```

### Full Command Sequence

```bash
# 1. Navigate to project
cd /Users/sankar/sankar/courses/agentic-ai/patient_advocacy_agent

# 2. Activate environment (optional)
source .venv/bin/activate

# 3. Download dataset
uv run python download_scin_dataset.py
# Follow prompts, takes ~20 minutes

# 4. Train embedder
uv run python train_embedder.py
# Takes ~2 hours on MPS, ~4-8 hours on CPU

# 5. Build index
uv run python build_index.py
# Takes ~15 minutes

# 6. Test everything
uv run python verify_setup.py
uv run python example_usage.py

# Done! 🎉
```

---

## Directory Structure After Completion

```
patient_advocacy_agent/
├── data/
│   └── scin/
│       ├── images/              ← 10,000+ skin images
│       └── metadata.csv         ← Image labels and metadata
│
├── models/
│   ├── embedder/
│   │   ├── checkpoints/         ← Training checkpoints
│   │   │   ├── embedder_epoch_1.pt
│   │   │   ├── embedder_epoch_2.pt
│   │   │   └── embedder_epoch_best.pt
│   │   └── final/
│   │       ├── embedder.pt      ← Fine-tuned model
│   │       ├── config.json
│   │       └── training_history.json
│   │
│   ├── similarity_index/        ← FAISS index
│   │   ├── faiss_index.bin
│   │   └── metadata.csv
│   │
│   └── rag_pipeline/            ← RAG components
│       ├── case_retriever/
│       └── knowledge_base/
│
└── reports/                     ← Generated assessments
```

---

## Validation

### Check Dataset

```bash
# Verify images exist
ls -la data/scin/images/ | head -20

# Check metadata
head -10 data/scin/metadata.csv
```

### Check Embedder

```bash
# Verify model was trained
ls -la models/embedder/final/

# Check training history
uv run python -c "
import json
with open('models/embedder/final/training_history.json') as f:
    history = json.load(f)
    print(f'Train losses: {len(history[\"train_losses\"])} epochs')
    print(f'Final loss: {history[\"train_losses\"][-1]:.4f}')
"
```

### Check Index

```bash
# Verify index was built
ls -la models/similarity_index/

# Check index summary
uv run python -c "
import json
with open('models/index_summary.json') as f:
    summary = json.load(f)
    print(f'Total images: {summary[\"dataset\"][\"total_images\"]}')
    print(f'Embeddings size: {summary[\"embeddings\"][\"size_mb\"]:.1f} MB')
"
```

---

## Performance Optimization

### For Faster Training

```python
# Use MPS (Apple Silicon)
device = 'mps'

# Larger batch size (if memory allows)
batch_size = 48

# More workers (for CPU)
num_workers = 4  # Only if using CPU, set to 0 for MPS
```

### For Faster Inference

```python
# Pre-cache embeddings (done by build_index.py)
# No additional actions needed
```

### For Faster Index Building

```python
# Already optimized
# FAISS automatically uses best available hardware
```

---

## Troubleshooting

### Dataset Download Issues

**Problem**: Can't download from GitHub
**Solution**:
```bash
# Download manually from browser
# Then run: uv run python download_scin_dataset.py
# And select: "Do you have the dataset ready? (y/n): y"
```

### Training Crashes

**Problem**: "CUDA out of memory" or similar
**Solution**:
```python
# In train_embedder.py
batch_size = 8  # Reduce batch size
num_workers = 0  # Always 0 for MPS

# Or limit epochs for testing
num_epochs = 2  # Test with 2 epochs first
```

### Index Building Slow

**Problem**: Takes too long
**Solution**: This is normal (10-20 minutes for 10K images)
```bash
# You can reduce dataset size for testing
# Or just wait for completion
```

---

## Next Steps

Once the pipeline is complete:

1. **Run Assessments**: `uv run python example_usage.py`
2. **Deploy API**: `uv run python -m uvicorn api:app`
3. **Fine-tune Further**: Retrain with more data
4. **Add Knowledge**: Add medical documents to RAG

---

## References

- SCIN Dataset: https://github.com/ISMAE-SUDA/SCIN
- SigLIP: https://huggingface.co/google/siglip-base-patch16-224
- FAISS: https://github.com/facebookresearch/faiss
- Patient Advocacy Agent: See README.md

---

**Status**: Complete guide for data pipeline
**Last Updated**: 2024
**Tested on**: Apple Silicon (M1/M2/M3)
