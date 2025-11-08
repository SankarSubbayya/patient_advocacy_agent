#!/usr/bin/env python
"""
Evaluate the hierarchical SigLIP model's retrieval accuracy.
This script tests how well the model can match images with their correct text descriptions
from a set of distractors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
import json
from tqdm import tqdm

from transformers import AutoModel, AutoProcessor

print("="*80)
print("HIERARCHICAL MODEL RETRIEVAL EVALUATION")
print("="*80)

# Configuration
class Config:
    data_dir = Path('/home/sankar/data/scin')
    metadata_file = data_dir / 'coarse_labeled_metadata_with_labels.csv'
    images_dir = data_dir / 'images'

    # Model paths
    model_dir = Path('/home/sankar/models/siglip_hierarchical/final_model')

    # Evaluation settings
    n_samples = 100  # Number of test samples to evaluate
    n_distractors = 9  # Number of distractor texts (total choices = 1 correct + 9 distractors)
    batch_size = 10  # Process in batches for efficiency

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

print(f"\nConfiguration:")
print(f"  Model: {config.model_dir}")
print(f"  Device: {config.device}")
print(f"  Test samples: {config.n_samples}")
print(f"  Choices per sample: {config.n_distractors + 1}")

# Load data
print("\nLoading data...")
df = pd.read_csv(config.metadata_file)
test_df = df[df['split'] == 'test'].reset_index(drop=True)
print(f"  Test set size: {len(test_df)} samples")

# Model class (same as training)
class SigLIPContrastiveModel(nn.Module):
    """SigLIP model for contrastive learning."""

    def __init__(self, model_name: str, temperature: float = 0.07):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_dict=True
        )

        image_embeds = outputs.vision_model_output.pooler_output
        text_embeds = outputs.text_model_output.pooler_output

        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()

        return {
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'image_embeds': image_embeds,
            'text_embeds': text_embeds,
        }

# Load model and processor
print("\nLoading model...")
processor = AutoProcessor.from_pretrained(config.model_dir)

# Load the model checkpoint directly with safetensors
from safetensors.torch import load_file

# Create our model wrapper
model = SigLIPContrastiveModel('google/siglip-base-patch16-224')

# Load the saved weights
checkpoint_path = config.model_dir / 'model.safetensors'
if checkpoint_path.exists():
    state_dict = load_file(checkpoint_path)

    # Filter out the logit_scale if it exists (we'll handle it separately)
    model_state = {k: v for k, v in state_dict.items() if not k.endswith('logit_scale')}

    # Load the main model weights
    missing_keys, unexpected_keys = model.model.load_state_dict(model_state, strict=False)
    print(f"  Loaded model weights from safetensors")

    # Load logit_scale if it exists
    if 'logit_scale' in state_dict:
        model.logit_scale.data = state_dict['logit_scale']
        print(f"  Loaded custom logit_scale: {model.logit_scale.item():.4f}")
    elif 'model.logit_scale' in state_dict:
        model.logit_scale.data = state_dict['model.logit_scale']
        print(f"  Loaded custom logit_scale: {model.logit_scale.item():.4f}")

model.to(config.device)
model.eval()
print("  Model loaded and set to evaluation mode")

def create_hierarchical_text(row):
    """Create hierarchical text description matching training format."""
    # Use consistent format for evaluation (matching validation/test format from training)
    text = f"{row['coarse_category']}: {row['condition']} - {row['description']}"
    return text

def evaluate_retrieval(model, processor, test_df, config):
    """Evaluate image-text retrieval accuracy."""

    # Sample test indices
    n_samples = min(config.n_samples, len(test_df))
    sample_indices = np.random.choice(len(test_df), n_samples, replace=False)

    accuracies = []
    top5_accuracies = []

    print(f"\nEvaluating {n_samples} samples...")
    for idx in tqdm(sample_indices, desc="Processing samples"):
        row = test_df.iloc[idx]

        # Load image
        image_path = config.images_dir / row['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            continue  # Skip if image can't be loaded

        # Create correct text
        correct_text = create_hierarchical_text(row)

        # Get distractor texts from other samples with different conditions
        other_indices = [i for i in range(len(test_df))
                        if i != idx and test_df.iloc[i]['condition'] != row['condition']]
        distractor_indices = np.random.choice(
            other_indices,
            min(config.n_distractors, len(other_indices)),
            replace=False
        )
        distractor_texts = [create_hierarchical_text(test_df.iloc[i])
                           for i in distractor_indices]

        # All text options (correct is always first)
        all_texts = [correct_text] + distractor_texts

        # Process batch
        inputs = processor(
            text=all_texts,
            images=[image] * len(all_texts),
            padding=True,
            truncation=True,
            max_length=64,  # Match training setting
            return_tensors="pt"
        ).to(config.device)

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            similarities = outputs['logits_per_image'][0]  # Image to all texts

        # Check accuracy
        sorted_indices = similarities.argsort(descending=True).cpu().numpy()

        # Top-1 accuracy
        predicted_idx = sorted_indices[0]
        correct = predicted_idx == 0  # Correct text is always at index 0
        accuracies.append(correct)

        # Top-5 accuracy
        top5_correct = 0 in sorted_indices[:5]
        top5_accuracies.append(top5_correct)

        # Show first few examples
        if len(accuracies) <= 3:
            print(f"\nExample {len(accuracies)}:")
            print(f"  Category: {row['coarse_category']}")
            print(f"  Condition: {row['condition']}")
            print(f"  Correct text: {correct_text[:80]}...")
            print(f"  Top-1 prediction: {'✓ Correct' if correct else '✗ Wrong'}")
            if not correct:
                print(f"    Selected: {all_texts[predicted_idx][:80]}...")
            print(f"  Top-5: {'✓' if top5_correct else '✗'}")
            print(f"  Similarity scores: {similarities.cpu().numpy()}")

    # Calculate metrics
    accuracy = np.mean(accuracies) if accuracies else 0
    top5_acc = np.mean(top5_accuracies) if top5_accuracies else 0

    return {
        'retrieval_accuracy': accuracy,
        'top5_accuracy': top5_acc,
        'n_evaluated': len(accuracies)
    }

# Run evaluation
print("\n" + "="*80)
print("Running Retrieval Evaluation")
print("="*80)

results = evaluate_retrieval(model, processor, test_df, config)

print("\n" + "="*80)
print("RETRIEVAL RESULTS")
print("="*80)
print(f"\n✓ Evaluated {results['n_evaluated']} samples")
print(f"  Top-1 Retrieval Accuracy: {results['retrieval_accuracy']:.1%}")
print(f"  Top-5 Retrieval Accuracy: {results['top5_accuracy']:.1%}")
print(f"\nBaseline (random): {1/(config.n_distractors + 1):.1%}")
print(f"Performance ratio: {results['retrieval_accuracy'] / (1/(config.n_distractors + 1)):.1f}x better than random")

# Save results
output_file = config.model_dir.parent / 'retrieval_evaluation.json'
with open(output_file, 'w') as f:
    json.dump({
        'model': 'hierarchical',
        'retrieval_accuracy': float(results['retrieval_accuracy']),
        'top5_accuracy': float(results['top5_accuracy']),
        'n_samples': results['n_evaluated'],
        'n_choices': config.n_distractors + 1,
        'random_baseline': 1/(config.n_distractors + 1)
    }, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")

# Compare with contrastive model if available
contrastive_summary = Path('/home/sankar/models/siglip_contrastive/training_summary.json')
if contrastive_summary.exists():
    with open(contrastive_summary, 'r') as f:
        contrastive_results = json.load(f)

    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"\nRetrieval Accuracy (10-way classification):")
    print(f"  Contrastive Model: {contrastive_results['retrieval_accuracy']:.1%}")
    print(f"  Hierarchical Model: {results['retrieval_accuracy']:.1%}")

    if results['retrieval_accuracy'] > contrastive_results['retrieval_accuracy']:
        improvement = (results['retrieval_accuracy'] - contrastive_results['retrieval_accuracy']) / contrastive_results['retrieval_accuracy']
        print(f"\n✅ Hierarchical model is {improvement:.0%} better!")
    else:
        print(f"\n📊 Contrastive model performs better on retrieval")

print("\n" + "="*80)
print("Evaluation complete!")