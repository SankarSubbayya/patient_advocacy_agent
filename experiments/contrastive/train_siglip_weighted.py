#!/usr/bin/env python
"""
Fine-tune SigLIP with WEIGHTED LOSS to handle class imbalance.

Improvements:
- Weighted CrossEntropyLoss based on class frequencies
- Balanced sampling option
- Per-class metrics tracking
- Better handling of minority classes
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional
from dataclasses import dataclass
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

from transformers import (
    AutoImageProcessor,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from torch import nn
from torch.utils.data import Dataset, WeightedRandomSampler
import evaluate

print("="*80)
print("SIGLIP FINE-TUNING WITH WEIGHTED LOSS")
print("="*80)
print("Improvements: Class-weighted loss to handle imbalance")
print("Dataset: SCIN with 16 coarse medical categories")
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
    num_labels: int = 16

    # Output
    output_dir: Path = Path('/home/sankar/models/siglip_weighted')

    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 30  # More epochs since we're using early stopping
    learning_rate: float = 3e-5  # Slightly lower for stability
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1

    # Training settings
    fp16: bool = torch.cuda.is_available()
    gradient_accumulation_steps: int = 2  # Effective batch size = 64
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "balanced_accuracy"  # Use balanced accuracy instead

    # Early stopping
    early_stopping_patience: int = 7  # More patience

    # Weighted sampling
    use_weighted_sampling: bool = False  # Set to True for balanced sampling

config = Config()
config.output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Model: {config.model_name}")
print(f"  Output: {config.output_dir}")
print(f"  Batch size: {config.batch_size} (effective: {config.batch_size * config.gradient_accumulation_steps})")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Epochs: {config.num_epochs}")
print(f"  Metric: {config.metric_for_best_model}")

# Load and analyze data
print("\n" + "="*80)
print("Loading and Analyzing Data")
print("="*80)

df = pd.read_csv(config.metadata_file)
print(f"Total samples: {len(df)}")

# Split data
train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df = df[df['split'] == 'val'].reset_index(drop=True)
test_df = df[df['split'] == 'test'].reset_index(drop=True)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df)}")
print(f"  Val: {len(val_df)}")
print(f"  Test: {len(test_df)}")

# Analyze class distribution
print("\nClass Distribution in Training Set:")
train_class_counts = Counter(train_df['coarse_category'])
for cat, count in sorted(train_class_counts.items()):
    pct = count / len(train_df) * 100
    print(f"  {cat:30s}: {count:4d} ({pct:5.1f}%)")

# Create label mapping
label_to_id = {cat: idx for idx, cat in enumerate(sorted(df['coarse_category'].unique()))}
id_to_label = {idx: cat for cat, idx in label_to_id.items()}

# Compute class weights for loss function
print("\n" + "="*80)
print("Computing Class Weights")
print("="*80)

# Get training labels
train_labels = train_df['coarse_category'].map(label_to_id).values

# Compute balanced class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# Convert to tensor
class_weights_tensor = torch.FloatTensor(class_weights)

print("\nClass weights (inverse frequency):")
for idx, weight in enumerate(class_weights):
    cat = id_to_label[idx]
    count = train_class_counts.get(cat, 0)
    print(f"  {idx:2d}. {cat:30s} ({count:4d} samples): weight={weight:.2f}")

# Compute sample weights for weighted sampling (optional)
if config.use_weighted_sampling:
    sample_weights = np.array([class_weights[label] for label in train_labels])
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    print(f"\n✓ Computed sample weights for balanced sampling")
else:
    sample_weights = None
    print("\n✗ Using standard random sampling (weighted loss only)")


# Custom dataset
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
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')

        # Process image
        encoding = self.image_processor(image, return_tensors='pt')
        pixel_values = encoding['pixel_values'].squeeze(0)

        # Get label
        label = self.label_to_id[row['coarse_category']]

        return {
            'pixel_values': pixel_values,
            'labels': label
        }


# Custom model with weighted loss
class SigLIPForImageClassification(nn.Module):
    """SigLIP model with weighted classification head."""

    def __init__(self, model_name: str, num_labels: int, class_weights=None):
        super().__init__()
        self.vision_model = AutoModel.from_pretrained(model_name).vision_model
        self.num_labels = num_labels
        self.class_weights = class_weights

        # Get hidden size
        hidden_size = self.vision_model.config.hidden_size

        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),  # Increased dropout
            nn.Linear(hidden_size, num_labels)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, pixel_values, labels=None):
        # Get vision features
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output

        # Classification
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # Use weighted cross entropy loss
            if self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
        }


# Load model and processor
print("\n" + "="*80)
print("Loading Model")
print("="*80)

image_processor = AutoImageProcessor.from_pretrained(config.model_name)
print(f"✓ Image processor loaded")

# Move class weights to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_weights_tensor = class_weights_tensor.to(device)

model = SigLIPForImageClassification(
    model_name=config.model_name,
    num_labels=config.num_labels,
    class_weights=class_weights_tensor
)
print(f"✓ Model loaded with weighted loss function")

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


# Custom metrics with per-class accuracy
print("\n" + "="*80)
print("Setting up Metrics")
print("="*80)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Compute overall and balanced accuracy."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Overall accuracy
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)

    # Per-class accuracy (balanced)
    per_class_acc = []
    for class_id in range(config.num_labels):
        class_mask = labels == class_id
        if class_mask.sum() > 0:
            class_acc = (predictions[class_mask] == class_id).mean()
            per_class_acc.append(class_acc)

    balanced_acc = np.mean(per_class_acc) if per_class_acc else 0.0

    return {
        'accuracy': accuracy['accuracy'],
        'balanced_accuracy': balanced_acc,
    }

print("✓ Metrics configured (accuracy + balanced accuracy)")


# Custom Trainer with weighted sampling (optional)
class WeightedTrainer(Trainer):
    """Trainer with optional weighted sampling."""

    def __init__(self, sample_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.sample_weights = sample_weights

    def get_train_dataloader(self):
        """Override to use weighted sampling if enabled."""
        if self.sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True
            )
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
            )
        else:
            return super().get_train_dataloader()


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
    # gradient_checkpointing=True,  # Disabled - not supported by custom model

    # Evaluation and saving
    eval_strategy=config.eval_strategy,
    save_strategy=config.save_strategy,
    load_best_model_at_end=config.load_best_model_at_end,
    metric_for_best_model=config.metric_for_best_model,
    greater_is_better=True,

    # Logging
    logging_dir=str(config.output_dir / 'logs'),
    logging_strategy="epoch",
    report_to="none",

    # Misc
    remove_unused_columns=False,
    dataloader_num_workers=0,
    save_total_limit=3,
    seed=42,
)

print("✓ Training arguments configured")
print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
print(f"  Total training steps: {len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps) * config.num_epochs}")


# Create trainer
print("\n" + "="*80)
print("Creating Trainer")
print("="*80)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=0.001  # Minimum improvement
        )
    ],
    sample_weights=sample_weights if config.use_weighted_sampling else None,
)

print(f"✓ Trainer created with:")
print(f"  - Weighted loss function (class weights)")
print(f"  - Balanced accuracy metric")
print(f"  - Early stopping (patience={config.early_stopping_patience})")
if config.use_weighted_sampling:
    print(f"  - Weighted sampling for balanced batches")


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


# Evaluate on test set with per-class metrics
print("\n" + "="*80)
print("Evaluating on Test Set")
print("="*80)

test_results = trainer.evaluate(test_dataset)

print("\nOverall Test Results:")
for key, value in test_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")

# Compute per-class accuracy
print("\n" + "="*80)
print("Per-Class Test Accuracy")
print("="*80)

# Get predictions
test_predictions = trainer.predict(test_dataset)
test_preds = np.argmax(test_predictions.predictions, axis=1)
test_labels = test_predictions.label_ids

# Calculate per-class metrics
for class_id in range(config.num_labels):
    class_name = id_to_label[class_id]
    class_mask = test_labels == class_id

    if class_mask.sum() > 0:
        class_acc = (test_preds[class_mask] == class_id).mean()
        class_support = class_mask.sum()
        print(f"  {class_id:2d}. {class_name:30s}: {class_acc:6.2%} (n={class_support})")
    else:
        print(f"  {class_id:2d}. {class_name:30s}: No samples in test set")


# Save final model
print("\n" + "="*80)
print("Saving Model")
print("="*80)

final_model_dir = config.output_dir / 'final_model'
trainer.save_model(str(final_model_dir))
image_processor.save_pretrained(str(final_model_dir))

print(f"✓ Model saved to: {final_model_dir}")

# Save label mapping and class weights
import json
metadata_path = config.output_dir / 'training_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump({
        'label_to_id': label_to_id,
        'id_to_label': id_to_label,
        'class_weights': class_weights.tolist(),
        'class_counts': dict(train_class_counts),
        'test_results': {k: float(v) if isinstance(v, (int, float)) else str(v)
                        for k, v in test_results.items()}
    }, f, indent=2)
print(f"✓ Training metadata saved to: {metadata_path}")


# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n✓ Training complete with weighted loss")
print(f"  Overall test accuracy: {test_results.get('eval_accuracy', 0):.2%}")
print(f"  Balanced test accuracy: {test_results.get('eval_balanced_accuracy', 0):.2%}")
print(f"\nModel saved to: {final_model_dir}")
print("\nImprovements applied:")
print("  • Weighted CrossEntropyLoss to handle class imbalance")
print("  • Balanced accuracy metric for better evaluation")
print("  • Per-class accuracy reporting")
print("  • Gradient accumulation for larger effective batch size")
print("  • More training epochs with early stopping")