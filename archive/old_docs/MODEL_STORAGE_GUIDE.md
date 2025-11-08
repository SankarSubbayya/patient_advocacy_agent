# Fine-Tuned Models Storage Guide

## Overview

This guide explains where fine-tuned models are stored, organized, and managed in the patient advocacy agent system.

---

## Default Storage Locations

### Base Directory Structure

By default, models are stored relative to your project root:

```
patient_advocacy_agent/
├── models/                          # ← Models stored here
│   ├── embedder/
│   │   ├── checkpoints/
│   │   │   ├── embedder_epoch_1.pt
│   │   │   ├── embedder_epoch_2.pt
│   │   │   └── embedder_epoch_best.pt
│   │   ├── final/
│   │   │   └── embedder.pt
│   │   └── processor.json           # SigLIP processor config
│   │
│   ├── similarity_index/
│   │   ├── faiss_index.bin          # FAISS index (binary)
│   │   ├── metadata.csv             # Image metadata
│   │   └── index_config.json
│   │
│   ├── rag_pipeline/
│   │   ├── case_retriever/
│   │   │   ├── case_vector_store/   # FAISS vector store
│   │   │   ├── metadata.csv
│   │   │   └── embeddings.npy
│   │   │
│   │   └── knowledge_base/
│   │       ├── vector_store/        # Medical docs FAISS
│   │       └── documents.json
│   │
│   ├── clustering/
│   │   ├── kmeans_model.pkl
│   │   └── cluster_labels.npy
│   │
│   └── agent/
│       └── agent_config.json
│
├── reports/                         # Generated reports
│   ├── assessments/
│   │   ├── PAT001/
│   │   │   ├── assessment_20240101_120000.json
│   │   │   └── assessment_20240102_150000.json
│   │   └── PAT002/
│   │       └── assessment_20240101_140000.json
│   │
│   └── reports/
│       ├── PAA-PAT001-20240101120000.json
│       ├── PAA-PAT001-20240102150000.json
│       └── PAA-PAT002-20240101140000.json
│
└── checkpoints/                     # Training checkpoints
    ├── embedder_epoch_1/
    ├── embedder_epoch_5/
    └── embedder_epoch_10_best/
```

---

## Storage Configuration

### 1. Setting Model Storage Path

You can configure where models are stored in two ways:

#### Option A: Environment Variable

```bash
# In .env file
export MODEL_DIR=/path/to/models
export REPORT_DIR=/path/to/reports
export CHECKPOINT_DIR=/path/to/checkpoints
```

#### Option B: Code Configuration

```python
from pathlib import Path
from patient_advocacy_agent import SigLIPEmbedder, EmbedderTrainer

# Define custom paths
MODEL_DIR = Path("/custom/path/models")
CHECKPOINT_DIR = Path("/custom/path/checkpoints")

# Create embedder
embedder = SigLIPEmbedder()

# Create trainer with custom checkpoint directory
trainer = EmbedderTrainer(embedder)

# Train with custom path
history = trainer.fit(
    train_loader,
    val_loader,
    num_epochs=10,
    checkpoint_dir=CHECKPOINT_DIR / "embedder"
)

# Save final model
embedder.save(MODEL_DIR / "embedder" / "final" / "embedder.pt")
```

---

## Model Types & Storage Locations

### 1. Image Embedder Models

**Purpose**: Fine-tuned SigLIP model for skin condition images

**Storage Path**: `models/embedder/`

**Files**:
```
models/embedder/
├── checkpoints/                     # Training checkpoints
│   ├── embedder_epoch_1.pt         # Checkpoint after epoch 1
│   ├── embedder_epoch_2.pt         # Checkpoint after epoch 2
│   ├── embedder_epoch_5.pt
│   ├── embedder_epoch_10.pt
│   └── embedder_epoch_10_best.pt   # Best checkpoint
│
└── final/
    ├── embedder.pt                 # Final trained model
    └── embedder_metadata.json       # Model config & metrics
```

**Size**: ~350-500 MB (Base), ~1.2-1.5 GB (Large)

**How to Save**:
```python
from patient_advocacy_agent import SigLIPEmbedder

embedder = SigLIPEmbedder()
# ... training code ...

# Save checkpoint during training (automatic)
embedder.save(Path("./models/embedder/checkpoints/embedder_epoch_10.pt"))

# Save final model
embedder.save(Path("./models/embedder/final/embedder.pt"))
```

**How to Load**:
```python
from patient_advocacy_agent import SigLIPEmbedder

# Load fine-tuned embedder
embedder = SigLIPEmbedder.load(Path("./models/embedder/final/embedder.pt"))

# Use for inference
embedding = embedder.extract_image_features(image_tensor)
```

---

### 2. Similarity Index (FAISS)

**Purpose**: Pre-computed similarity index for fast case retrieval

**Storage Path**: `models/similarity_index/`

**Files**:
```
models/similarity_index/
├── faiss_index.bin               # FAISS index file (binary)
├── metadata.csv                  # Image metadata with labels
├── embeddings.npy                # Pre-computed embeddings (optional)
└── index_config.json             # Index configuration
```

**Size**:
- FAISS Index: ~40 MB per 10K images
- Metadata: ~500 KB per 10K images
- Embeddings: ~20 MB per 10K images

**How to Save**:
```python
from patient_advocacy_agent import SimilarityIndex
import numpy as np

# Create and save index
index = SimilarityIndex(embeddings, metadata_df, use_gpu=True)
index.save(Path("./models/similarity_index"))

# Files automatically created:
# - faiss_index.bin
# - metadata.csv
```

**How to Load**:
```python
from patient_advocacy_agent import SimilarityIndex

# Load index
index = SimilarityIndex.load(Path("./models/similarity_index"), use_gpu=True)

# Use for search
similar_cases = index.search(query_embedding, k=5)
```

---

### 3. RAG Pipeline

**Purpose**: Vector stores for medical knowledge and case retrieval

**Storage Path**: `models/rag_pipeline/`

**Structure**:
```
models/rag_pipeline/
├── case_retriever/
│   ├── case_vector_store/        # FAISS vector store (LangChain)
│   │   ├── index.faiss
│   │   ├── index.pkl
│   │   └── docstore.pkl
│   ├── metadata.csv
│   └── embeddings.npy
│
└── knowledge_base/
    ├── vector_store/             # Medical docs vector store
    │   ├── index.faiss
    │   ├── index.pkl
    │   └── docstore.pkl
    └── documents.json
```

**Size**:
- Case Vector Store: ~50 MB per 10K cases
- Knowledge Base: Variable (depends on documents)

**How to Save**:
```python
from patient_advocacy_agent import RAGPipeline, CaseRetriever, MedicalKnowledgeBase

# Create components
retriever = CaseRetriever(metadata_df, embeddings)
kb = MedicalKnowledgeBase()
kb.add_documents(medical_docs)

# Create pipeline
rag = RAGPipeline(retriever, kb)

# Save entire pipeline
rag.save(Path("./models/rag_pipeline"))
```

**How to Load**:
```python
from patient_advocacy_agent import RAGPipeline

# Load RAG pipeline
rag = RAGPipeline.load(Path("./models/rag_pipeline"))

# Use for retrieval
context = rag.retrieve_context(condition, symptoms)
```

---

### 4. Clustering Models

**Purpose**: K-means clustering for embedding analysis

**Storage Path**: `models/clustering/`

**Files**:
```
models/clustering/
├── kmeans_model.pkl              # Scikit-learn KMeans model
├── cluster_labels.npy            # Pre-computed cluster labels
└── clustering_config.json        # Configuration
```

**Size**: ~10 MB per model

**How to Save**:
```python
from patient_advocacy_agent import ImageClusterer

# Create and fit clusterer
clusterer = ImageClusterer(n_clusters=20)
labels = clusterer.fit(embeddings)

# Save model
clusterer.save(Path("./models/clustering/kmeans_model.pkl"))
```

**How to Load**:
```python
from patient_advocacy_agent import ImageClusterer

# Load clusterer
clusterer = ImageClusterer.load(Path("./models/clustering/kmeans_model.pkl"))

# Predict on new embeddings
new_labels = clusterer.predict(new_embeddings)
```

---

## Complete Model Training & Storage Workflow

### Step 1: Prepare Data

```python
from pathlib import Path
from patient_advocacy_agent import SCINDataLoader

# Setup directories
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "scin"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "embedder" / "checkpoints"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
data_loader = SCINDataLoader(data_dir=DATA_DIR, batch_size=32)
dataloaders = data_loader.create_dataloaders()
```

### Step 2: Fine-tune Embedder

```python
from patient_advocacy_agent import SigLIPEmbedder, EmbedderTrainer
import torch

# Create embedder
embedder = SigLIPEmbedder(
    model_name="google/siglip-base-patch16-224",
    projection_dim=512
)

# Create trainer
trainer = EmbedderTrainer(
    embedder=embedder,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    learning_rate=1e-4
)

# Train with checkpointing
history = trainer.fit(
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
    num_epochs=20,
    checkpoint_dir=CHECKPOINT_DIR,
    early_stopping_patience=3
)

# Save final model
embedder.save(MODEL_DIR / "embedder" / "final" / "embedder.pt")

print(f"✓ Model saved to {MODEL_DIR / 'embedder' / 'final' / 'embedder.pt'}")
```

### Step 3: Extract Embeddings

```python
import numpy as np

# Extract embeddings for all images
all_embeddings = []
all_metadata = []

for batch in dataloaders['train']:  # Use all data
    images = batch['image'].to(embedder.device)
    with torch.no_grad():
        embeddings = embedder.extract_image_features(images).cpu().numpy()
    all_embeddings.append(embeddings)

embeddings = np.concatenate(all_embeddings).astype(np.float32)
print(f"✓ Extracted {len(embeddings)} embeddings")
```

### Step 4: Build Similarity Index

```python
from patient_advocacy_agent import SimilarityIndex

# Create index
index = SimilarityIndex(
    embeddings=embeddings,
    metadata_df=metadata_df,
    use_gpu=torch.cuda.is_available()
)

# Save index
index.save(MODEL_DIR / "similarity_index")

print(f"✓ Similarity index saved to {MODEL_DIR / 'similarity_index'}")
```

### Step 5: Setup RAG Pipeline

```python
from patient_advocacy_agent import CaseRetriever, MedicalKnowledgeBase, RAGPipeline
from langchain.schema import Document

# Create case retriever
case_retriever = CaseRetriever(metadata_df, embeddings)

# Create knowledge base
kb = MedicalKnowledgeBase()

# Add medical documents
medical_docs = [
    Document(page_content="Eczema info...", metadata={"condition": "eczema"}),
    Document(page_content="Psoriasis info...", metadata={"condition": "psoriasis"}),
    # ... more documents
]
kb.add_documents(medical_docs)

# Create RAG pipeline
rag = RAGPipeline(case_retriever, kb)

# Save RAG
rag.save(MODEL_DIR / "rag_pipeline")

print(f"✓ RAG pipeline saved to {MODEL_DIR / 'rag_pipeline'}")
```

### Step 6: Save Complete Configuration

```python
import json

# Create training summary
config = {
    "training_date": datetime.now().isoformat(),
    "model_paths": {
        "embedder": str(MODEL_DIR / "embedder" / "final" / "embedder.pt"),
        "similarity_index": str(MODEL_DIR / "similarity_index"),
        "rag_pipeline": str(MODEL_DIR / "rag_pipeline"),
    },
    "model_specs": {
        "embedder_model": "google/siglip-base-patch16-224",
        "projection_dim": 512,
        "num_embeddings": len(embeddings),
        "num_epochs": 20,
        "batch_size": 32,
    },
    "metrics": {
        "final_train_loss": history['train_losses'][-1],
        "final_val_loss": history['val_losses'][-1],
    }
}

# Save config
config_path = MODEL_DIR / "training_config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Training config saved to {config_path}")
```

---

## Organizing Models by Version

### Recommended Directory Structure for Multiple Versions

```
models/
├── v1.0_initial/                   # First trained model
│   ├── embedder/
│   │   └── final/
│   │       └── embedder.pt
│   ├── similarity_index/
│   ├── rag_pipeline/
│   └── training_config.json
│
├── v1.1_improved/                  # Improved with more data
│   ├── embedder/
│   │   └── final/
│   │       └── embedder.pt
│   ├── similarity_index/
│   ├── rag_pipeline/
│   └── training_config.json
│
├── v2.0_large_model/               # Using larger SigLIP variant
│   ├── embedder/
│   │   └── final/
│   │       └── embedder.pt
│   ├── similarity_index/
│   ├── rag_pipeline/
│   └── training_config.json
│
└── production/                      # Active production model
    ├── embedder -> v2.0_large_model/embedder
    ├── similarity_index -> v2.0_large_model/similarity_index
    ├── rag_pipeline -> v2.0_large_model/rag_pipeline
    └── training_config.json
```

**How to manage versions**:

```python
from pathlib import Path
import shutil

def save_model_version(model_dir, version_name):
    """Save current models with version name"""
    version_dir = model_dir.parent / version_name
    version_dir.mkdir(exist_ok=True)

    for component in ['embedder', 'similarity_index', 'rag_pipeline']:
        src = model_dir / component
        dst = version_dir / component
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

def load_model_version(model_root, version_name):
    """Load models from specific version"""
    version_dir = model_root / version_name
    return {
        'embedder': version_dir / 'embedder',
        'index': version_dir / 'similarity_index',
        'rag': version_dir / 'rag_pipeline'
    }

# Save version
save_model_version(Path("./models"), "v1.0_initial")

# Load version
paths = load_model_version(Path("./models"), "v1.0_initial")
```

---

## Model Registry (Optional)

For production deployments, use a model registry:

```python
import json
from datetime import datetime

class ModelRegistry:
    """Track all trained models and their metadata"""

    def __init__(self, registry_path):
        self.registry_path = registry_path
        self.registry = self._load_registry()

    def _load_registry(self):
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {"models": []}

    def register_model(self, name, version, path, metrics):
        """Register a trained model"""
        entry = {
            "name": name,
            "version": version,
            "path": str(path),
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "status": "active"
        }
        self.registry["models"].append(entry)
        self._save_registry()

    def get_latest(self, name):
        """Get latest version of a model"""
        models = [m for m in self.registry["models"] if m["name"] == name]
        return max(models, key=lambda x: x["timestamp"]) if models else None

    def _save_registry(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

# Usage
registry = ModelRegistry(Path("./models/registry.json"))

registry.register_model(
    name="embedder",
    version="1.0",
    path=Path("./models/embedder/final/embedder.pt"),
    metrics={
        "val_loss": 0.234,
        "train_accuracy": 0.95,
        "inference_time_ms": 20
    }
)

# Load latest
latest = registry.get_latest("embedder")
print(f"Latest embedder: {latest['path']}")
```

---

## Storage Best Practices

### 1. Use Symbolic Links for Production

```bash
# Create symlink to current production model
ln -s ./models/v2.0_large_model/embedder ./models/production/embedder
```

### 2. Backup Important Models

```bash
# Backup to external storage
tar -czf embedder_backup_$(date +%Y%m%d).tar.gz ./models/embedder/final/

# Upload to cloud
aws s3 cp embedder_backup_*.tar.gz s3://my-bucket/backups/
```

### 3. Monitor Model Size

```python
def get_model_size(path):
    """Get total size of model directory"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024**3)  # Convert to GB

size_gb = get_model_size(Path("./models"))
print(f"Total model size: {size_gb:.2f} GB")
```

### 4. Version Control (Git LFS)

```bash
# Install Git LFS
brew install git-lfs

# Track large files
git lfs track "models/**/*.pt"
git lfs track "models/**/*.bin"

# Commit as normal
git add .
git commit -m "Add trained models"
```

---

## Troubleshooting

### Model Not Found

```python
# Check if model exists
model_path = Path("./models/embedder/final/embedder.pt")

if not model_path.exists():
    print(f"✗ Model not found at {model_path}")
    print("Available models:")
    for f in model_path.parent.parent.rglob("*.pt"):
        print(f"  - {f}")
else:
    embedder = SigLIPEmbedder.load(model_path)
    print(f"✓ Model loaded successfully")
```

### Out of Disk Space

```bash
# Check disk usage
du -sh ./models/

# Clean old checkpoints
find ./models/embedder/checkpoints -name "*.pt" -mtime +30 -delete

# Compress old versions
tar -czf models_v1.0.tar.gz ./models/v1.0_initial/
rm -rf ./models/v1.0_initial/
```

### Loading Wrong Model Version

```python
# Always check model config
config = torch.load(model_path, map_location='cpu')
print(f"Model version: {config.get('version', 'unknown')}")
print(f"Training date: {config.get('training_date', 'unknown')}")
```

---

## Summary

**Default Locations:**
- **Embedder**: `models/embedder/final/embedder.pt`
- **Index**: `models/similarity_index/faiss_index.bin`
- **RAG**: `models/rag_pipeline/`
- **Reports**: `reports/assessments/` and `reports/reports/`

**Size Estimates:**
- Embedder: 350-500 MB (Base) or 1.2-1.5 GB (Large)
- Index: ~40 MB per 10K images
- RAG: ~50 MB per 10K cases
- **Total**: ~600 MB to 2 GB depending on data size

**Best Practices:**
1. Version your models
2. Keep training configs
3. Use Git LFS for version control
4. Backup production models
5. Clean up old checkpoints
6. Monitor disk usage
