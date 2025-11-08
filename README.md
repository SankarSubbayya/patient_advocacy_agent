# Patient Advocacy Agent

SigLIP vision-language model for dermatology, trained on SCIN dataset (66 skin conditions).

## 🚀 Quick Start

```python
from patient_advocacy_agent import VisionLanguageEmbedder

# Load best model (20% accuracy)
embedder = VisionLanguageEmbedder(
    model_path="/home/sankar/models/siglip_fine_grained/final_model"
)

# Use the model
image_emb = embedder.embed_image("skin_image.jpg")
text_emb = embedder.embed_text("Patient with psoriasis")
```

## 📊 Results

| Model | Accuracy | vs Baseline |
|-------|----------|-------------|
| **Fine-Grained** (66 classes) | **20%** | **2x** |
| Contrastive (16 classes) | 20% | 2x |
| Hierarchical | 13% | 1.3x |

*Baseline: 10% (random)*

## 📁 Structure

```
experiments/    # Training scripts
utils/         # Data tools
src/           # Core code
config.yaml    # Settings
```

## 🔧 Train

```bash
python experiments/fine_grained/train_siglip_fine_grained.py
```

## 📚 Docs

- [Integration](claude_integration_example.py)
- [Data Policy](DATA_SHARING.md)
- [Full Details](PROJECT_STRUCTURE.md)
