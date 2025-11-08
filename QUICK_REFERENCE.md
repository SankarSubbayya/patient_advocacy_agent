# Quick Reference

## Commands

```bash
# Training
python experiments/fine_grained/train_siglip_fine_grained.py

# Evaluation
python experiments/hierarchical/evaluate_hierarchical_retrieval.py

# Analysis
python experiments/analysis/plot_loss_text.py

# Cleanup
python cleanup_project.py
```

## Key Paths

- **Best Model**: `/home/sankar/models/siglip_fine_grained/final_model`
- **Config**: `config.yaml`
- **Training Scripts**: `experiments/`
- **Data Utils**: `utils/`

## Model Stats

- **Accuracy**: 20% (2x baseline)
- **Classes**: 66 skin conditions
- **Training Loss**: 81.9% reduction
- **Architecture**: SigLIP (Google)

## Python Usage

```python
# Load model
from patient_advocacy_agent import VisionLanguageEmbedder
embedder = VisionLanguageEmbedder(model_path="...")

# Embed
img_emb = embedder.embed_image("image.jpg")
txt_emb = embedder.embed_text("description")

# Find similar
results = embedder.find_similar(img_emb, top_k=5)
```

## Conditions Covered

16 Categories, 66 Specific Conditions including:
- Eczema, Psoriasis, Dermatitis
- Acne, Folliculitis, Impetigo
- Tinea, Urticaria, Alopecia
- [Full list](COARSE_CATEGORIES_README.md)