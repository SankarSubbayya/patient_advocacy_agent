#!/usr/bin/env python
"""
Train SigLIP with FINE-GRAINED contrastive learning.
Uses the 66 specific condition labels with their descriptions for more precise matching.

This approach should provide:
- Better discrimination between specific conditions
- More precise retrieval accuracy
- Direct condition-level matching without hierarchical complexity
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
print("SIGLIP FINE-GRAINED CONTRASTIVE LEARNING")
print("="*80)
print("Using 66 fine-grained condition labels for precise matching")
print("Direct condition-level contrastive learning")
print("="*80)

@dataclass
class Config:
    # Data paths
    data_dir: Path = Path('/home/sankar/data/scin')
    metadata_file: Path = data_dir / 'coarse_labeled_metadata_with_labels.csv'
    images_dir: Path = data_dir / 'images'

    # Model
    model_name: str = 'google/siglip-base-patch16-224'
    output_dir: Path = Path('/home/sankar/models/siglip_fine_grained')

    # Training
    batch_size: int = 32
    num_epochs: int = 25
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 500

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16: bool = torch.cuda.is_available()
    temperature: float = 0.07

config = Config()
config.output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Model: {config.model_name}")
print(f"  Output: {config.output_dir}")
print(f"  Device: {config.device}")
print(f"  Batch size: {config.batch_size}")
print(f"  Epochs: {config.num_epochs}")

# Load data
print("\n" + "="*80)
print("Loading Data")
print("="*80)

df = pd.read_csv(config.metadata_file)
print(f"Total samples: {len(df)}")
print(f"Fine-grained conditions: {df['condition'].nunique()}")
print(f"Coarse categories: {df['coarse_category'].nunique()}")

# Show condition distribution
print("\nTop 10 most common conditions:")
condition_counts = df['condition'].value_counts()
for condition, count in condition_counts.head(10).items():
    pct = count / len(df) * 100
    print(f"  {condition:35s}: {count:4d} ({pct:4.1f}%)")

print("\n10 rarest conditions:")
for condition, count in condition_counts.tail(10).items():
    pct = count / len(df) * 100
    print(f"  {condition:35s}: {count:4d} ({pct:4.1f}%)")

# Split data
train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df = df[df['split'] == 'val'].reset_index(drop=True)
test_df = df[df['split'] == 'test'].reset_index(drop=True)

print(f"\nSplits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


class FineGrainedContrastiveDataset(Dataset):
    """Dataset for fine-grained contrastive learning."""

    def __init__(self, df: pd.DataFrame, images_dir: Path, processor, mode='train'):
        self.df = df
        self.images_dir = images_dir
        self.processor = processor
        self.mode = mode

        # Templates focusing on specific conditions
        self.templates = {
            'condition_focused': [
                "{condition}",
                "Patient with {condition}",
                "Clinical image of {condition}",
                "Diagnosed as {condition}",
                "{condition}: {description}",
            ],
            'detailed': [
                "{condition} - {description}",
                "Medical condition: {condition}. {description}",
                "{description} (Diagnosis: {condition})",
            ],
            'category_hint': [
                "{condition} ({coarse})",
                "{condition}, a type of {coarse}",
                "{coarse} condition: {condition}",
            ]
        }

    def create_fine_grained_text(self, row):
        """Create text focusing on fine-grained condition."""
        condition = row['condition'].lower()
        description = row['description']
        coarse = row['coarse_category'].lower()

        if self.mode == 'train':
            # Varied templates for training
            template_type = np.random.choice(list(self.templates.keys()), p=[0.4, 0.4, 0.2])
            template = np.random.choice(self.templates[template_type])

            text = template.format(
                condition=condition,
                description=description,
                coarse=coarse
            )

            # Sometimes add medical context
            if np.random.random() > 0.7:
                prefixes = [
                    "Dermatological finding: ",
                    "Skin examination reveals: ",
                    "Clinical presentation: ",
                    "Visual diagnosis: ",
                ]
                text = np.random.choice(prefixes) + text

        else:
            # Consistent format for validation/test
            text = f"{row['condition']}: {description}"

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

        # Create fine-grained text
        text = self.create_fine_grained_text(row)

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
train_dataset = FineGrainedContrastiveDataset(train_df, config.images_dir, processor, mode='train')
val_dataset = FineGrainedContrastiveDataset(val_df, config.images_dir, processor, mode='val')
test_dataset = FineGrainedContrastiveDataset(test_df, config.images_dir, processor, mode='test')

print(f"✓ Datasets created")
print(f"  Train: {len(train_dataset)} samples")
print(f"  Val: {len(val_dataset)} samples")
print(f"  Test: {len(test_dataset)} samples")

# Show sample texts
print("\nSample fine-grained texts:")
for i in range(3):
    sample_text = train_dataset.create_fine_grained_text(train_df.iloc[i])
    condition = train_df.iloc[i]['condition']
    print(f"  [{condition}] {sample_text[:80]}...")


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
print("STARTING FINE-GRAINED TRAINING")
print("="*80)
print("Training with fine-grained condition labels...")
print("Focus on discriminating between 66 specific conditions")
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


# Test retrieval
print("\n" + "="*80)
print("Testing Fine-Grained Retrieval")
print("="*80)

def test_retrieval(model, processor, test_df, images_dir, n_samples=20):
    """Test fine-grained retrieval capabilities."""
    model.eval()

    sample_indices = np.random.choice(len(test_df), min(n_samples, len(test_df)), replace=False)

    accuracies = []
    condition_accuracies = {}

    for idx in tqdm(sample_indices, desc="Testing retrieval"):
        row = test_df.iloc[idx]

        # Load image
        image_path = images_dir / row['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            continue

        # Create text options - use different conditions
        correct_text = f"{row['condition']}: {row['description']}"
        correct_condition = row['condition']

        # Get distractors from different conditions
        other_conditions = test_df[test_df['condition'] != correct_condition]['condition'].unique()
        distractor_conditions = np.random.choice(
            other_conditions,
            min(9, len(other_conditions)),
            replace=False
        )

        distractor_texts = []
        for dist_cond in distractor_conditions:
            dist_row = test_df[test_df['condition'] == dist_cond].iloc[0]
            distractor_texts.append(f"{dist_row['condition']}: {dist_row['description']}")

        all_texts = [correct_text] + distractor_texts

        # Process
        inputs = processor(
            text=all_texts,
            images=[image] * len(all_texts),
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(config.device)

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            similarities = outputs['logits_per_image'][0]

        # Check accuracy
        predicted_idx = similarities.argmax().item()
        correct = predicted_idx == 0
        accuracies.append(correct)

        # Track per-condition accuracy
        if correct_condition not in condition_accuracies:
            condition_accuracies[correct_condition] = []
        condition_accuracies[correct_condition].append(correct)

    accuracy = np.mean(accuracies)
    print(f"\n✓ Fine-grained retrieval accuracy: {accuracy:.2%}")

    # Show per-condition performance
    print("\nPer-condition accuracy (samples with data):")
    for condition, accs in sorted(condition_accuracies.items()):
        if len(accs) >= 1:
            cond_acc = np.mean(accs)
            print(f"  {condition:35s}: {cond_acc:6.2%} (n={len(accs)})")

    return accuracy

retrieval_acc = test_retrieval(model, processor, test_df, config.images_dir)

# Save summary
summary = {
    'approach': 'fine_grained',
    'num_conditions': int(df['condition'].nunique()),
    'train_samples': len(train_df),
    'final_loss': float(train_result.training_loss),
    'test_loss': float(test_results['eval_loss']),
    'retrieval_accuracy': float(retrieval_acc),
}

with open(config.output_dir / 'training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("FINE-GRAINED TRAINING COMPLETE")
print("="*80)
print("\nAdvantages of fine-grained approach:")
print("  • Direct discrimination between 66 specific conditions")
print("  • No hierarchical complexity")
print("  • Focused on exact condition matching")
print("  • Should improve retrieval accuracy")
print(f"\nFinal metrics:")
print(f"  Test loss: {test_results['eval_loss']:.4f}")
print(f"  Retrieval accuracy: {retrieval_acc:.2%}")