#!/usr/bin/env python
"""
Train SigLIP with PROPER CONTRASTIVE LEARNING using text descriptions.

SigLIP is a vision-language model that learns by matching images with text.
This script uses the actual text descriptions from the metadata for proper
contrastive learning, not just classification.
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
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

print("="*80)
print("SIGLIP CONTRASTIVE LEARNING WITH TEXT DESCRIPTIONS")
print("="*80)
print("Using vision-language contrastive learning (the proper way!)")
print("Matching images with their medical text descriptions")
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

    # Output
    output_dir: Path = Path('/home/sankar/models/siglip_contrastive')

    # Training
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 500

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16: bool = torch.cuda.is_available()

    # Temperature for contrastive loss
    temperature: float = 0.07

config = Config()
config.output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Model: {config.model_name}")
print(f"  Device: {config.device}")
print(f"  Batch size: {config.batch_size}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Temperature: {config.temperature}")

# Load data
print("\n" + "="*80)
print("Loading Data")
print("="*80)

df = pd.read_csv(config.metadata_file)
print(f"Total samples: {len(df)}")

# Check description column
print("\nSample descriptions:")
for i in range(5):
    desc = df.iloc[i]['description']
    cat = df.iloc[i]['coarse_category']
    print(f"  {i+1}. [{cat}] {desc}")

# Split data
train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df = df[df['split'] == 'val'].reset_index(drop=True)
test_df = df[df['split'] == 'test'].reset_index(drop=True)

print(f"\nSplits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


class SigLIPContrastiveDataset(Dataset):
    """Dataset for SigLIP contrastive learning with text descriptions."""

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        processor,
        augment_text: bool = True
    ):
        self.df = df
        self.images_dir = images_dir
        self.processor = processor
        self.augment_text = augment_text

        # Text templates for augmentation
        self.templates = [
            "{}",  # Original description
            "A medical image showing {}",
            "Clinical photograph of {}",
            "Dermatological image: {}",
            "Patient presenting with {}",
            "Skin condition: {}",
            "Diagnostic image of {}",
        ]

    def __len__(self):
        return len(self.df)

    def augment_description(self, description: str, category: str) -> str:
        """Augment text description for better learning."""
        if self.augment_text and np.random.random() > 0.5:
            # Sometimes use category-based description
            if np.random.random() > 0.5:
                template = np.random.choice(self.templates)
                return template.format(category.lower())
            else:
                # Add category to original description
                return f"{description} - {category}"
        return description

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image_path = self.images_dir / row['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')

        # Get text description
        description = row['description']
        category = row['coarse_category']

        # Augment text for training
        if hasattr(self, 'augment_text'):
            text = self.augment_description(description, category)
        else:
            text = description

        # Process image and text together
        inputs = self.processor(
            text=text,
            images=image,
            padding="max_length",
            truncation=True,
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
        # Get image and text embeddings
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Normalize embeddings
        image_embeds = outputs.vision_model_output.pooler_output
        text_embeds = outputs.text_model_output.pooler_output

        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Compute similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()

        # Compute contrastive loss
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


# Load processor and model
print("\n" + "="*80)
print("Loading Model and Processor")
print("="*80)

processor = AutoProcessor.from_pretrained(config.model_name)
print("✓ Processor loaded")

model = SigLIPContrastiveModel(
    model_name=config.model_name,
    temperature=config.temperature
)
model.to(config.device)
print(f"✓ Model loaded and moved to {config.device}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# Create datasets
print("\n" + "="*80)
print("Creating Datasets")
print("="*80)

train_dataset = SigLIPContrastiveDataset(
    train_df, config.images_dir, processor, augment_text=True
)
val_dataset = SigLIPContrastiveDataset(
    val_df, config.images_dir, processor, augment_text=False
)
test_dataset = SigLIPContrastiveDataset(
    test_df, config.images_dir, processor, augment_text=False
)

print(f"✓ Train dataset: {len(train_dataset)} samples (with text augmentation)")
print(f"✓ Val dataset: {len(val_dataset)} samples")
print(f"✓ Test dataset: {len(test_dataset)} samples")

# Test one sample
print("\nTesting data loading...")
sample = train_dataset[0]
print(f"  Sample keys: {sample.keys()}")
print(f"  Pixel values shape: {sample['pixel_values'].shape}")
print(f"  Input IDs shape: {sample['input_ids'].shape}")


# Custom trainer for contrastive learning
class ContrastiveTrainer(Trainer):
    """Custom trainer for vision-language contrastive learning."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute contrastive loss."""
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

        # Return image-text similarity scores as predictions
        return (loss, outputs['logits_per_image'].cpu(), None)


# Training arguments
print("\n" + "="*80)
print("Configuring Training")
print("="*80)

training_args = TrainingArguments(
    output_dir=str(config.output_dir),

    # Training
    num_train_epochs=config.num_epochs,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,

    # Optimization
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
    warmup_steps=config.warmup_steps,
    fp16=config.fp16,

    # Evaluation
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,

    # Saving
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,

    # Other
    remove_unused_columns=False,
    dataloader_num_workers=0,
    report_to="none",
    seed=42,
)

print("✓ Training arguments configured")
print(f"  Total training steps: ~{len(train_dataset) // config.batch_size * config.num_epochs}")

# Create trainer
trainer = ContrastiveTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("✓ Trainer created")

# Train
print("\n" + "="*80)
print("STARTING CONTRASTIVE TRAINING")
print("="*80)
print("\nTraining with vision-language contrastive learning...")
print("Matching images with their medical text descriptions...")
print()

train_result = trainer.train()

print("\n" + "="*80)
print("Training Complete")
print("="*80)
print(f"Final training loss: {train_result.training_loss:.4f}")

# Evaluate
print("\n" + "="*80)
print("Evaluating on Test Set")
print("="*80)

test_results = trainer.evaluate(test_dataset)
print(f"Test loss: {test_results['eval_loss']:.4f}")

# Save model
print("\n" + "="*80)
print("Saving Model")
print("="*80)

final_path = config.output_dir / 'final_model'
trainer.save_model(str(final_path))
processor.save_pretrained(str(final_path))

print(f"✓ Model saved to: {final_path}")

# Test the model with retrieval
print("\n" + "="*80)
print("Testing Image-Text Retrieval")
print("="*80)

def test_retrieval(model, processor, test_df, images_dir, n_samples=10):
    """Test image-text retrieval capabilities."""
    model.eval()

    # Sample random test images
    sample_indices = np.random.choice(len(test_df), min(n_samples, len(test_df)), replace=False)

    accuracies = []
    for idx in sample_indices:
        row = test_df.iloc[idx]

        # Load image
        image_path = images_dir / row['image_path']
        image = Image.open(image_path).convert('RGB')

        # Create text options (correct description + distractors)
        correct_text = row['description']
        correct_category = row['coarse_category']

        # Get distractor texts from other samples
        other_indices = [i for i in range(len(test_df)) if i != idx]
        distractor_indices = np.random.choice(other_indices, min(9, len(other_indices)), replace=False)
        distractor_texts = [test_df.iloc[i]['description'] for i in distractor_indices]

        # All text options
        all_texts = [correct_text] + distractor_texts

        # Process
        inputs = processor(
            text=all_texts,
            images=[image] * len(all_texts),
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(config.device)

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            similarities = outputs['logits_per_image'][0]  # Image to all texts

        # Check if correct text has highest similarity
        predicted_idx = similarities.argmax().item()
        correct = predicted_idx == 0
        accuracies.append(correct)

        if len(accuracies) <= 3:  # Show first few examples
            print(f"\nExample {len(accuracies)}:")
            print(f"  Category: {correct_category}")
            print(f"  Correct text: {correct_text[:100]}...")
            print(f"  Predicted: {'✓ Correct' if correct else '✗ Wrong'}")
            if not correct:
                print(f"  Selected: {all_texts[predicted_idx][:100]}...")

    accuracy = np.mean(accuracies)
    print(f"\n✓ Retrieval accuracy on {len(accuracies)} samples: {accuracy:.2%}")
    return accuracy

# Test retrieval
retrieval_acc = test_retrieval(model, processor, test_df, config.images_dir)

# Save training summary
summary = {
    'model': config.model_name,
    'training_samples': len(train_df),
    'batch_size': config.batch_size,
    'learning_rate': config.learning_rate,
    'num_epochs': config.num_epochs,
    'final_train_loss': float(train_result.training_loss),
    'test_loss': float(test_results['eval_loss']),
    'retrieval_accuracy': float(retrieval_acc),
}

with open(config.output_dir / 'training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\n✓ SigLIP trained with proper contrastive learning")
print(f"  Used text descriptions for vision-language matching")
print(f"  Final test loss: {test_results['eval_loss']:.4f}")
print(f"  Retrieval accuracy: {retrieval_acc:.2%}")
print(f"\nModel can now:")
print("  • Match images with medical descriptions")
print("  • Retrieve similar cases based on text queries")
print("  • Generate meaningful embeddings for both images and text")