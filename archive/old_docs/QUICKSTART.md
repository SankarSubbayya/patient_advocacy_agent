# Patient Advocacy Agent - Quick Start Guide

This guide will help you get started with the patient advocacy agent system for dermatology assessment.

## Overview

The patient advocacy agent is a comprehensive system that:

1. **Fine-tunes SigLIP embeddings** on skin condition images using contrastive learning
2. **Clusters images** to find visually similar cases
3. **Retrieves medical knowledge** using RAG (Retrieval-Augmented Generation)
4. **Generates assessments** with the MedGemini agent
5. **Creates physician reports** with evidence-based recommendations

## Installation

### Prerequisites

- Python 3.12+
- CUDA 11.8+ (optional, for GPU support)

### Setup

```bash
# Clone the repository
cd patient_advocacy_agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Quick Start: 5-Minute Example

### Step 1: Load the Data

```python
from pathlib import Path
from patient_advocacy_agent import SCINDataLoader

# Initialize data loader
data_loader = SCINDataLoader(
    data_dir=Path("/path/to/scin/dataset"),
    batch_size=32
)

# Prepare dataset (assumes metadata.csv exists)
dataloaders = data_loader.create_dataloaders()

# Access train/val/test splits
train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']
```

### Step 2: Fine-tune SigLIP Embedder

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

# Train embedder
history = trainer.fit(
    train_loader,
    val_loader,
    num_epochs=10,
    checkpoint_dir=Path("./checkpoints")
)

# Save embedder
embedder.save(Path("./models/embedder.pt"))
```

### Step 3: Build Similarity Index

```python
from patient_advocacy_agent import SimilarityIndex
import numpy as np

# Extract embeddings for all training images
embeddings = []
metadata_records = []

for batch in train_loader:
    images = batch['image'].to(embedder.device)
    with torch.no_grad():
        emb = embedder.extract_image_features(images).cpu().numpy()
    embeddings.append(emb)
    metadata_records.extend(batch)

embeddings = np.concatenate(embeddings)

# Create similarity index
index = SimilarityIndex(
    embeddings=embeddings,
    metadata_df=train_df,
    use_gpu=torch.cuda.is_available()
)

# Save index
index.save(Path("./models/similarity_index"))
```

### Step 4: Set Up RAG System

```python
from patient_advocacy_agent import (
    CaseRetriever,
    MedicalKnowledgeBase,
    RAGPipeline
)
from langchain.schema import Document

# Create case retriever
case_retriever = CaseRetriever(
    metadata_df=train_df,
    embeddings=embeddings
)

# Create knowledge base
kb = MedicalKnowledgeBase()

# Add documents (would typically load from medical literature)
# For demo, create sample documents
sample_docs = [
    Document(
        page_content="Eczema (atopic dermatitis) is a chronic inflammatory skin condition...",
        metadata={'source': 'medical_database', 'condition': 'eczema'}
    ),
    Document(
        page_content="Psoriasis is a chronic autoimmune condition affecting skin...",
        metadata={'source': 'medical_database', 'condition': 'psoriasis'}
    ),
]
kb.add_documents(sample_docs)

# Create RAG pipeline
rag_pipeline = RAGPipeline(case_retriever, kb)

# Save RAG system
rag_pipeline.save(Path("./models/rag_pipeline"))
```

### Step 5: Create and Use the Agent

```python
from patient_advocacy_agent import (
    MedGeminiAgent,
    PatientCase,
    PatientAssessmentAPI,
    PatientAssessmentRequest
)

# Create agent
agent = MedGeminiAgent(
    embedder=embedder,
    rag_pipeline=rag_pipeline,
    clustering_index=index
)

# Create API
api = PatientAssessmentAPI(
    agent=agent,
    embedder=embedder,
    storage_dir=Path("./reports")
)

# Create assessment request
request = PatientAssessmentRequest(
    patient_id="P001",
    age=35,
    gender="F",
    symptoms=["itching", "redness", "dryness"],
    symptom_onset="2 weeks ago",
    patient_notes="Recently started new skincare product"
)

# Run assessment
result = api.assess_patient(request)

print(f"Status: {result['status']}")
print(f"Top condition: {result['assessment']['suspected_conditions'][0]['name']}")
print(f"Confidence: {result['assessment']['suspected_conditions'][0]['confidence']:.1%}")
```

### Step 6: Generate Physician Report

```python
# Generate report
report_result = api.generate_physician_report("P001")

if report_result['status'] == 'success':
    report = report_result['report']
    print(f"Report ID: {report['report_id']}")
    print(f"Similar cases found: {len(report['assessment']['similar_cases'])}")

    # Export report
    from patient_advocacy_agent import ReportExportRequest

    export_request = ReportExportRequest(
        report_id=report['report_id'],
        format="txt"
    )

    export_result = api.export_report(export_request)
    print(export_result['content'])
```

## Project Structure

```
patient_advocacy_agent/
├── src/patient_advocacy_agent/
│   ├── __init__.py              # Package initialization and exports
│   ├── data.py                  # SCIN dataset loading and preprocessing
│   ├── embedder.py              # SigLIP embedder and contrastive loss
│   ├── clustering.py            # Similarity indexing and clustering
│   ├── rag.py                   # RAG pipeline for knowledge retrieval
│   ├── agent.py                 # MedGemini agent for assessment
│   └── api.py                   # REST API interface
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── pyproject.toml               # Project dependencies
└── README.md                    # Detailed documentation
```

## Key Components

### 1. Data Pipeline (`data.py`)
- `SkinConditionDataset`: PyTorch dataset for skin images
- `SCINDataLoader`: Manages loading and splitting SCIN data
- `ImageMetadata`: Structured metadata for images

### 2. Image Embedder (`embedder.py`)
- `SigLIPEmbedder`: Vision-language model for image embeddings
- `ContrastiveLoss`: Pulls similar conditions together, pushes different ones apart
- `EmbedderTrainer`: Training loop with early stopping

### 3. Similarity & Clustering (`clustering.py`)
- `SimilarityIndex`: FAISS-based fast similarity search
- `ImageClusterer`: K-means clustering of embeddings
- `ConditionBasedGrouping`: Group images by skin condition

### 4. RAG System (`rag.py`)
- `MedicalKnowledgeBase`: Vector store for medical documents
- `CaseRetriever`: Retrieve similar historical cases
- `RAGPipeline`: Combined retrieval system

### 5. Assessment Agent (`agent.py`)
- `MedGeminiAgent`: Main assessment engine
- `PatientCase`: Patient information structure
- `AssessmentResult`: Assessment output
- `PhysicianReport`: Formatted report for physicians

### 6. API Interface (`api.py`)
- `PatientAssessmentAPI`: REST-ready API
- `PatientAssessmentRequest`: Request validation
- `ReportExportRequest`: Export format specification

## Configuration

Set environment variables in `.env`:

```bash
# OpenAI API (for future LLM integration)
export OPENAI_API_KEY=your-key

# Data paths
export SCIN_DATA_DIR=/path/to/scin/dataset

# Model paths
export MODEL_DIR=./models
export REPORT_DIR=./reports
```

## Next Steps

1. **Download SCIN Dataset**: Get the Skin Condition Image Network dataset
2. **Prepare Data**: Create metadata.csv with image paths and labels
3. **Fine-tune Embedder**: Run training on your GPU
4. **Test Assessment**: Try the API with sample patient cases
5. **Deploy**: Set up REST API for clinical use

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `SCINDataLoader`
- Use gradient accumulation in trainer
- Use `faiss-cpu` instead of GPU FAISS

### Slow Similarity Search
- Reduce number of clusters in `SimilarityIndex`
- Use IndexFlatL2 for smaller datasets
- Pre-compute and cache embeddings

### Missing Medical Knowledge
- Add documents to `MedicalKnowledgeBase` via `add_documents()`
- Use medical literature PDFs
- Integrate with external knowledge bases

## References

- **SigLIP**: https://huggingface.co/google/siglip-base-patch16-224
- **SCIN Dataset**: https://github.com/ISMAE-SUDA/SCIN
- **FAISS**: https://github.com/facebookresearch/faiss
- **LangChain**: https://github.com/langchain-ai/langchain

## Support

For issues or questions:
1. Check the documentation in `/docs`
2. Review example notebooks in `/docs/notebooks`
3. Check existing issues on GitHub
4. Submit new issues with detailed reproduction steps

---

**Version**: 0.1.0
**Last Updated**: 2024
**Maintainer**: SupportVectors AI Lab
