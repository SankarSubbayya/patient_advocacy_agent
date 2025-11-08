#!/usr/bin/env python
"""
Fine-tune SigLIP on SCIN dataset using Hugging Face Transformers.

This script uses:
- Hugging Face Trainer API for robust training
- Coarse-grained labels (16 categories)
- Standard image classification approach
- Mixed precision training for efficiency
"""

import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

from transformers import (
    AutoImageProcessor,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from torch import nn
from torch.utils.data import Dataset
import evaluate

print("="*80)
print("SIGLIP FINE-TUNING WITH HUGGING FACE")
print("="*80)
print("Dataset: SCIN with 16 coarse medical categories")
print("Model: google/siglip-base-patch16-224")
print("Framework: Hugging Face Transformers")
print("="*80)

# Configuration
@dataclass
class Config:
    # Data paths
    data_dir: Path = Path('/home/sankar/data/scin')
    metadata_file: Path = data_dir / 'coarse_labeled_metadata_with_labels.csv'
    images_dir: Path = data_dir / 'images'

    # Model
    model_name: str = 'google/siglip-base-patch16-224'
    num_labels: int = 16  # Coarse categories

    # Output
    output_dir: Path = Path('/home/sankar/models/siglip_hf_coarse')

    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.1

    # Training settings
    fp16: bool = torch.cuda.is_available()  # Mixed precision if GPU available
    gradient_accumulation_steps: int = 1
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"

    # Early stopping
    early_stopping_patience: int = 5

config = Config()
config.output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Model: {config.model_name}")
print(f"  Output: {config.output_dir}")
print(f"  Batch size: {config.batch_size}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Epochs: {config.num_epochs}")
print(f"  Mixed precision (fp16): {config.fp16}")


# Load and verify data
print("\n" + "="*80)
print("Loading Data")
print("="*80)

df = pd.read_csv(config.metadata_file)
print(f"Total samples: {len(df)}")
print(f"Coarse categories: {df['coarse_category'].nunique()}")

# Verify splits
train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df = df[df['split'] == 'val'].reset_index(drop=True)
test_df = df[df['split'] == 'test'].reset_index(drop=True)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df)}")
print(f"  Val: {len(val_df)}")
print(f"  Test: {len(test_df)}")

# Show category distribution
print("\nCategory distribution:")
for cat in sorted(df['coarse_category'].unique()):
    count = (df['coarse_category'] == cat).sum()
    pct = count / len(df) * 100
    print(f"  {cat:30s}: {count:4d} ({pct:5.1f}%)")

# Create label mapping
label_to_id = {cat: idx for idx, cat in enumerate(sorted(df['coarse_category'].unique()))}
id_to_label = {idx: cat for cat, idx in label_to_id.items()}

print(f"\nLabel mapping ({len(label_to_id)} classes):")
for cat, idx in sorted(label_to_id.items(), key=lambda x: x[1]):
    print(f"  {idx:2d}: {cat}")


# Create custom dataset
class SCINDataset(Dataset):
    """SCIN dataset for Hugging Face."""

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        image_processor,
        label_to_id: Dict[str, int]
    ):
        self.df = df
        self.images_dir = images_dir
        self.image_processor = image_processor
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image_path = self.images_dir / row['image_path']
        image = Image.open(image_path).convert('RGB')

        # Process image
        encoding = self.image_processor(image, return_tensors='pt')

        # Remove batch dimension
        pixel_values = encoding['pixel_values'].squeeze(0)

        # Get label
        label = self.label_to_id[row['coarse_category']]

        return {
            'pixel_values': pixel_values,
            'labels': label
        }


# Custom SigLIP model for classification
class SigLIPForImageClassification(nn.Module):
    """SigLIP model with classification head."""

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.vision_model = AutoModel.from_pretrained(model_name).vision_model
        self.num_labels = num_labels

        # Get hidden size from vision model config
        hidden_size = self.vision_model.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, pixel_values, labels=None):
        # Get vision features
        outputs = self.vision_model(pixel_values=pixel_values)

        # Use pooled output (CLS token)
        pooled_output = outputs.pooler_output

        # Classification
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
        }


# Load image processor and model
print("\n" + "="*80)
print("Loading Model")
print("="*80)

image_processor = AutoImageProcessor.from_pretrained(config.model_name)
print(f"✓ Image processor loaded")

model = SigLIPForImageClassification(
    model_name=config.model_name,
    num_labels=config.num_labels
)
print(f"✓ Model loaded with classification head")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")


# Create datasets
print("\n" + "="*80)
print("Creating Datasets")
print("="*80)

train_dataset = SCINDataset(train_df, config.images_dir, image_processor, label_to_id)
val_dataset = SCINDataset(val_df, config.images_dir, image_processor, label_to_id)
test_dataset = SCINDataset(test_df, config.images_dir, image_processor, label_to_id)

print(f"✓ Train dataset: {len(train_dataset)} samples")
print(f"✓ Val dataset: {len(val_dataset)} samples")
print(f"✓ Test dataset: {len(test_dataset)} samples")


# Metrics
print("\n" + "="*80)
print("Setting up Metrics")
print("="*80)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Compute accuracy and loss."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)

    return {
        'accuracy': accuracy['accuracy'],
    }

print("✓ Metrics configured (accuracy)")


# Training arguments
print("\n" + "="*80)
print("Configuring Training")
print("="*80)

training_args = TrainingArguments(
    output_dir=str(config.output_dir),

    # Training hyperparameters
    num_train_epochs=config.num_epochs,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
    warmup_ratio=config.warmup_ratio,

    # Optimization
    fp16=config.fp16,
    gradient_accumulation_steps=config.gradient_accumulation_steps,

    # Evaluation and saving
    eval_strategy=config.eval_strategy,
    save_strategy=config.save_strategy,
    load_best_model_at_end=config.load_best_model_at_end,
    metric_for_best_model=config.metric_for_best_model,
    greater_is_better=True,

    # Logging
    logging_dir=str(config.output_dir / 'logs'),
    logging_strategy="epoch",
    report_to="none",  # Disable wandb/tensorboard

    # Misc
    remove_unused_columns=False,
    dataloader_num_workers=0,
    save_total_limit=3,  # Keep only 3 best checkpoints
)

print("✓ Training arguments configured")
print(f"  Total training steps: {len(train_dataset) // config.batch_size * config.num_epochs}")
print(f"  Warmup steps: {int(len(train_dataset) // config.batch_size * config.num_epochs * config.warmup_ratio)}")


# Create trainer
print("\n" + "="*80)
print("Creating Trainer")
print("="*80)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
    ],
)

print("✓ Trainer created with early stopping callback")


# Train
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print()

train_result = trainer.train()

print("\n" + "="*80)
print("Training Complete")
print("="*80)
print(f"Training loss: {train_result.training_loss:.4f}")
print(f"Training time: {train_result.metrics['train_runtime']:.1f}s")


# Evaluate on test set
print("\n" + "="*80)
print("Evaluating on Test Set")
print("="*80)

test_results = trainer.evaluate(test_dataset)

print("\nTest Results:")
for key, value in test_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")


# Save final model
print("\n" + "="*80)
print("Saving Model")
print("="*80)

final_model_dir = config.output_dir / 'final_model'
trainer.save_model(str(final_model_dir))
image_processor.save_pretrained(str(final_model_dir))

print(f"✓ Model saved to: {final_model_dir}")

# Save label mapping
import json
label_mapping_path = config.output_dir / 'label_mapping.json'
with open(label_mapping_path, 'w') as f:
    json.dump({
        'label_to_id': label_to_id,
        'id_to_label': id_to_label,
    }, f, indent=2)
print(f"✓ Label mapping saved to: {label_mapping_path}")


# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nBest model saved to: {final_model_dir}")
print(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
print(f"\nTraining complete! Next steps:")
print("  1. Run clustering analysis with the fine-tuned model")
print("  2. Compare: Vanilla SigLIP vs Fine-tuned (211) vs Fine-tuned HF (16)")
print("  3. Use the best model for patient advocacy embeddings")
