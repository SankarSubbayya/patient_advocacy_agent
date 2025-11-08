#!/usr/bin/env python
"""
Train SigLIP with HIERARCHICAL text descriptions.
Uses both coarse categories (16) and fine-grained conditions (211) in the text.

This gives the best of both worlds:
- Semantic grouping from coarse categories
- Precise diagnosis from fine conditions
- Rich text descriptions combining both levels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import json

from transformers import (
    AutoModel,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset

print("="*80)
print("SIGLIP HIERARCHICAL CONTRASTIVE LEARNING")
print("="*80)
print("Using BOTH coarse categories AND fine-grained conditions in text")
print("Best of both worlds: semantic grouping + precise diagnosis")
print("="*80)

@dataclass
class Config:
    data_dir: Path = Path('/home/sankar/data/scin')
    metadata_file: Path = data_dir / 'coarse_labeled_metadata_with_labels.csv'
    images_dir: Path = data_dir / 'images'

    model_name: str = 'google/siglip-base-patch16-224'
    output_dir: Path = Path('/home/sankar/models/siglip_hierarchical')

    batch_size: int = 24  # Smaller due to longer text
    num_epochs: int = 25
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 500

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16: bool = torch.cuda.is_available()
    temperature: float = 0.07

config = Config()
config.output_dir.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(config.metadata_file)
print(f"\nTotal samples: {len(df)}")
print(f"Fine-grained conditions: {df['condition'].nunique()}")
print(f"Coarse categories: {df['coarse_category'].nunique()}")

# Split data
train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df = df[df['split'] == 'val'].reset_index(drop=True)
test_df = df[df['split'] == 'test'].reset_index(drop=True)

print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Show hierarchical structure
print("\nHierarchical structure examples:")
for i in range(5):
    row = df.iloc[i]
    print(f"  {row['coarse_category']:25s} → {row['condition']:30s}")


class HierarchicalTextDataset(Dataset):
    """Dataset with hierarchical text descriptions."""

    def __init__(self, df: pd.DataFrame, images_dir: Path, processor, mode='train'):
        self.df = df
        self.images_dir = images_dir
        self.processor = processor
        self.mode = mode

        # Templates for hierarchical descriptions
        self.templates = {
            'hierarchical': [
                "{coarse}: {condition} - {description}",
                "{description} ({condition}, type: {coarse})",
                "Category: {coarse}. Diagnosis: {condition}. {description}",
                "{coarse} condition presenting as {condition}. {description}",
            ],
            'medical': [
                "Medical image showing {condition}, a type of {coarse}",
                "Clinical presentation of {condition} ({coarse})",
                "Dermatological finding: {condition} in the {coarse} category",
                "Patient with {condition}, classified as {coarse}",
            ],
            'descriptive': [
                "{description}",  # Original description
                "{coarse}: {description}",
                "{condition} - {description}",
            ]
        }

    def create_hierarchical_text(self, row):
        """Create rich hierarchical text description."""
        coarse = row['coarse_category'].lower()
        condition = row['condition'].lower()
        description = row['description']

        # For training, use varied templates
        if self.mode == 'train':
            # Randomly choose template type
            template_type = np.random.choice(list(self.templates.keys()))
            template = np.random.choice(self.templates[template_type])

            # Add medical terminology variations
            if np.random.random() > 0.7:
                # Sometimes add more medical context
                medical_terms = {
                    'Inflammatory Dermatitis': 'erythematous and pruritic',
                    'Bacterial Infections': 'purulent with possible fever',
                    'Fungal Infections': 'with scaling and possible ring formation',
                    'Viral Infections': 'vesicular or papular eruption',
                    'Skin Cancer': 'irregular borders, possible ulceration',
                    'Urticaria/Allergic': 'wheals with surrounding erythema',
                }
                extra = medical_terms.get(row['coarse_category'], '')
                if extra:
                    description = f"{description}. Features: {extra}"

            text = template.format(
                coarse=coarse,
                condition=condition,
                description=description
            )
        else:
            # For validation/test, use consistent format
            text = f"{row['coarse_category']}: {row['condition']} - {description}"

        return text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image_path = self.images_dir / row['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')

        # Create hierarchical text
        text = self.create_hierarchical_text(row)

        # Process
        inputs = self.processor(
            text=text,
            images=image,
            padding="max_length",
            truncation=True,
            max_length=64,  # SigLIP uses 64 tokens max
            return_tensors="pt"
        )

        # Remove batch dimension
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)

        return inputs


# Model (same as before)
class SigLIPContrastiveModel(nn.Module):
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

        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=image_embeds.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        return {
            'loss': loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'image_embeds': image_embeds,
            'text_embeds': text_embeds,
        }


print("\n" + "="*80)
print("Loading Model and Creating Datasets")
print("="*80)

processor = AutoProcessor.from_pretrained(config.model_name)
model = SigLIPContrastiveModel(config.model_name, config.temperature)
model.to(config.device)

print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

# Create datasets
train_dataset = HierarchicalTextDataset(train_df, config.images_dir, processor, mode='train')
val_dataset = HierarchicalTextDataset(val_df, config.images_dir, processor, mode='val')
test_dataset = HierarchicalTextDataset(test_df, config.images_dir, processor, mode='test')

print(f"✓ Datasets created")

# Test sample texts
print("\nSample hierarchical texts:")
for i in range(3):
    sample_text = train_dataset.create_hierarchical_text(train_df.iloc[i])
    print(f"  {i+1}. {sample_text[:100]}...")


# Custom trainer
class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step for contrastive learning."""
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs['loss']

        if prediction_loss_only:
            return (loss, None, None)

        # Return loss and logits for metrics
        return (loss, outputs['logits_per_image'].cpu(), None)


# Training arguments
training_args = TrainingArguments(
    output_dir=str(config.output_dir),
    num_train_epochs=config.num_epochs,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
    warmup_steps=config.warmup_steps,
    fp16=config.fp16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    report_to="none",
)

trainer = ContrastiveTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("\n" + "="*80)
print("STARTING HIERARCHICAL TRAINING")
print("="*80)
print("Training with hierarchical text: coarse → fine → description")
print()

train_result = trainer.train()

print(f"\n✓ Training complete! Final loss: {train_result.training_loss:.4f}")

# Evaluate
test_results = trainer.evaluate(test_dataset)
print(f"✓ Test loss: {test_results['eval_loss']:.4f}")

# Save
final_path = config.output_dir / 'final_model'
trainer.save_model(str(final_path))
processor.save_pretrained(str(final_path))
print(f"✓ Model saved to: {final_path}")

# Save summary
summary = {
    'approach': 'hierarchical',
    'fine_conditions': int(df['condition'].nunique()),
    'coarse_categories': int(df['coarse_category'].nunique()),
    'train_samples': len(train_df),
    'final_loss': float(train_result.training_loss),
    'test_loss': float(test_results['eval_loss']),
}

with open(config.output_dir / 'training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("HIERARCHICAL TRAINING COMPLETE")
print("="*80)
print("\nAdvantages of this approach:")
print("  • Combines coarse semantic grouping with fine diagnosis")
print("  • Rich text with both levels of information")
print("  • Better handles class imbalance (coarse context helps rare conditions)")
print("  • Enables both triage (coarse) and diagnosis (fine) queries")