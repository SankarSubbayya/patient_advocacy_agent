# Coarse-Grained Category Training

## Problem

The initial fine-tuning attempt with **211 fine-grained skin conditions** degraded clustering performance compared to vanilla SigLIP:

| Metric | Vanilla SigLIP | Fine-tuned (211) | Winner |
|--------|---------------|------------------|--------|
| NMI | 0.3201 | 0.2454 | Vanilla ✓ |
| Homogeneity | 0.3824 | 0.2905 | Vanilla ✓ |
| Cluster Purity | 26.95% | 22.49% | Vanilla ✓ |

**Root cause**: Too many classes (211) with too few samples per class (~31 images/condition) makes contrastive learning ineffective.

## Solution: Coarse-Grained Grouping

Created **16 medically meaningful coarse categories** by grouping similar conditions:

### Category Mapping

1. **Inflammatory Dermatitis** (2,201 images, 37.2%)
   - Eczema, Contact Dermatitis, Seborrheic Dermatitis, etc.

2. **Urticaria/Allergic** (638 images, 10.8%)
   - Urticaria, Drug Rash, Hypersensitivity, Erythema multiforme

3. **Bacterial Infections** (547 images, 9.3%)
   - Impetigo, Folliculitis, Cellulitis, Abscess

4. **Parasitic/Insect** (445 images, 7.5%)
   - Insect Bite, Scabies, Flea dermatosis

5. **Viral Infections** (371 images, 6.3%)
   - Herpes Zoster, Herpes Simplex, Viral Exanthem, Warts

6. **Psoriatic Conditions** (329 images, 5.6%)
   - Psoriasis, Pityriasis rosea

7. **Acne/Follicular** (319 images, 5.4%)
   - Acne, Rosacea, Keratosis pilaris

8. **Fungal Infections** (283 images, 4.8%)
   - Tinea, Tinea Versicolor, Candida

9. **Vascular Disorders** (274 images, 4.6%)
   - Purpura, Vasculitis, Pigmented purpuric eruption

10. **Autoimmune/Lichenoid** (168 images, 2.8%)
    - Lichen planus, Lupus, Dermatomyositis

11. **Trauma/Wounds** (118 images, 2.0%)
    - Scars, Abrasions, Inflicted lesions

12. **Benign Tumors** (70 images, 1.2%)
    - Seborrheic Keratosis, Granuloma annulare

13. **Skin Cancer** (57 images, 1.0%)
    - BCC, SCC, Melanoma, Actinic Keratosis

14. **Pruritic Conditions** (54 images, 0.9%)
    - Prurigo nodularis

15. **Pigmentary Disorders** (29 images, 0.5%)
    - Hyperpigmentation, Vitiligo, Melasma

16. **Bullous Disorders** (6 images, 0.1%)
    - Bullous dermatitis, Pemphigus

**Total**: 5,909 labeled images (608 "Other" excluded)

## Expected Benefits

### Before (211 fine-grained)
- 6,517 images across 211 conditions
- Average: **~31 images per condition**
- Too sparse for effective contrastive learning
- Model learned to separate but not meaningfully

### After (16 coarse)
- 5,909 images across 16 categories
- Average: **~369 images per category** (12x more!)
- Simpler decision boundaries
- Better alignment with medical practice

### Why This Should Work

1. **More samples per class**: 369 vs 31 images
2. **Meaningful groupings**: Categories reflect actual diagnostic reasoning
3. **Balanced complexity**: Not too fine (211), not too coarse (1)
4. **Better gradient signal**: Sufficient negative pairs for contrastive loss

## Files Created

### Data Preparation
- `analyze_conditions.py` - Analyze distribution and create coarse mappings
- `create_coarse_metadata.py` - Generate final coarse-labeled dataset
- `/home/sankar/data/scin/coarse_labeled_metadata_with_labels.csv` - Final dataset
- `/home/sankar/data/scin/coarse_category_mapping.json` - Category to label mapping

### Training
- `train_embedder_coarse.py` - Training script for coarse categories

### Expected Outputs (after training)
- `/home/sankar/models/embedder_coarse_labels/final/embedder_coarse_labels.pt`
- Training history and curves

## Usage

### 1. Train with Coarse Categories

```bash
uv run python train_embedder_coarse.py
```

Expected: ~20 epochs, early stopping when validation loss plateaus

### 2. Compare All Three Models

Create comparison script to evaluate:
- **Vanilla SigLIP**: Baseline (no training)
- **Fine-tuned (211 classes)**: Previous attempt (degraded performance)
- **Fine-tuned (16 classes)**: New coarse-grained approach

Metrics:
- Silhouette Score (cluster quality)
- NMI, Homogeneity, Completeness (label alignment)
- Cluster Purity
- Per-category clustering analysis

### 3. Expected Results

Hypothesis: **Fine-tuned (16 classes) > Vanilla > Fine-tuned (211 classes)**

Reasoning:
- 16 classes provide enough samples for meaningful learning
- Coarse categories match medical decision-making
- Simpler task allows better optimization

## Next Steps After Training

1. Run three-way comparison (vanilla vs 211 vs 16)
2. If coarse performs well:
   - Use for patient advocacy embeddings
   - Consider hierarchical approach (coarse → fine-grained)
3. If still underperforms vanilla:
   - Investigate different loss functions (triplet, supervised contrastive)
   - Try different architectures (CLIP, DINOv2)
   - Collect more training data

## Medical Rationale

The coarse categories reflect how dermatologists actually think:
- First, identify the **type** of condition (inflammatory vs infectious vs neoplastic)
- Then, narrow down to specific diagnosis

This hierarchical structure makes the embedding space more useful:
- Similar conditions cluster together (all bacterial infections near each other)
- Clear separation between different types
- Interpretable and actionable for downstream tasks
