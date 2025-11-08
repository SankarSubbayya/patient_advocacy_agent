#!/usr/bin/env python
"""
Full training script for SigLIP embedder with labeled SCIN data.

This script:
- Uses synthetic_labeled_metadata.csv with 30 skin conditions
- Trains for 20 epochs with early stopping
- Uses optimized learning rate
- Saves checkpoints and final model
"""

import torch
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime

print("="*80)
print("FULL SIGLIP TRAINING WITH REAL MEDICAL LABELS")
print("="*80)
print("Using: real_labeled_metadata.csv (6517 images, 211 conditions)")
print("Labels: From expert dermatologists (not synthetic!)")
print("="*80)

# Configuration
class Config:
    # Paths
    data_dir = Path('/home/sankar/data/scin')
    metadata_file = data_dir / 'real_labeled_metadata.csv'  # NOW USING REAL LABELS!
    images_dir = data_dir / 'images'

    # Model output
    model_dir = Path('/home/sankar/models/embedder_real_labels')  # New directory for real labels
    checkpoint_dir = model_dir / 'checkpoints'
    final_dir = model_dir / 'final'

    # Training params
    batch_size = 32
    num_epochs = 20
    learning_rate = 5e-5  # Slightly higher for better learning
    weight_decay = 1e-5
    early_stopping_patience = 5

    # Model
    model_name = 'google/siglip-base-patch16-224'
    projection_dim = 512
    freeze_backbone = False  # Fine-tune entire model

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

# Create directories
config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
config.final_dir.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Device: {config.device}")
print(f"  Batch size: {config.batch_size}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Epochs: {config.num_epochs}")
print(f"  Model output: {config.model_dir}")

# Verify labeled data
print("\n" + "-"*80)
print("Verifying Labeled Data")
print("-"*80)

df = pd.read_csv(config.metadata_file)
print(f"Total samples: {len(df)}")
print(f"Unique conditions: {df['condition'].nunique()}")

conditions = df['condition'].value_counts()
print("\nTop 5 conditions:")
for condition, count in conditions.head(5).items():
    print(f"  {condition}: {count}")

# Import after verification
print("\n" + "-"*80)
print("Loading Modules")
print("-"*80)

from patient_advocacy_agent import (
    SCINDataLoader,
    SigLIPEmbedder,
    EmbedderTrainer
)

# Create data loader
print("\nCreating data loaders...")
data_loader = SCINDataLoader(
    data_dir=config.data_dir,
    batch_size=config.batch_size,
    num_workers=0  # Set to 0 for stability
)

# Use labeled metadata
dataloaders = data_loader.create_dataloaders(
    metadata_path=config.metadata_file,
    images_dir='images'
)

print(f"  Train batches: {len(dataloaders['train'])}")
print(f"  Val batches: {len(dataloaders['val'])}")
print(f"  Test batches: {len(dataloaders['test'])}")

# Create model
print("\nCreating SigLIP embedder...")
embedder = SigLIPEmbedder(
    model_name=config.model_name,
    projection_dim=config.projection_dim,
    freeze_backbone=config.freeze_backbone
)
print(f"  ✓ Model created")
print(f"  Moving to {config.device}...")
embedder.to(config.device)

# Create trainer
print("\nCreating trainer...")
trainer = EmbedderTrainer(
    embedder=embedder,
    device=config.device,
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay
)

# Add learning rate scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
trainer.scheduler = CosineAnnealingLR(
    trainer.optimizer,
    T_max=config.num_epochs,
    eta_min=1e-6
)
print("  ✓ Trainer created with cosine annealing scheduler")

# Training loop with monitoring
print("\n" + "="*80)
print("STARTING FULL TRAINING")
print("="*80)

best_val_loss = float('inf')
patience_counter = 0
training_start = datetime.now()

for epoch in range(config.num_epochs):
    print(f"\n[Epoch {epoch+1}/{config.num_epochs}]")

    # Train
    print("  Training...", end='', flush=True)
    train_loss = trainer.train_epoch(dataloaders['train'])
    print(f" Loss: {train_loss:.4f}")

    # Validate
    print("  Validating...", end='', flush=True)
    val_loss = trainer.validate(dataloaders['val'])
    print(f" Loss: {val_loss:.4f}")

    # Learning rate
    current_lr = trainer.optimizer.param_groups[0]['lr']
    print(f"  Learning rate: {current_lr:.2e}")

    # Step scheduler
    if trainer.scheduler:
        trainer.scheduler.step()

    # Checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        # Save checkpoint
        checkpoint_path = config.checkpoint_dir / f"best_model_epoch_{epoch+1}.pt"
        embedder.save(checkpoint_path)
        print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"  Patience: {patience_counter}/{config.early_stopping_patience}")

        if patience_counter >= config.early_stopping_patience:
            print("\n⚠ Early stopping triggered!")
            break

    # Show progress
    if epoch == 0:
        initial_train_loss = train_loss
        initial_val_loss = val_loss

    train_improvement = ((initial_train_loss - train_loss) / initial_train_loss) * 100
    val_improvement = ((initial_val_loss - val_loss) / initial_val_loss) * 100

    print(f"  Progress: Train improved {train_improvement:.1f}%, Val improved {val_improvement:.1f}%")

# Training complete
training_time = (datetime.now() - training_start).total_seconds() / 60
print(f"\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Training time: {training_time:.1f} minutes")
print(f"Final epoch: {epoch + 1}")
print(f"Best val loss: {best_val_loss:.4f}")

# Save final model and history
print("\nSaving final model...")
final_model_path = config.final_dir / "embedder_labeled_full.pt"
embedder.save(final_model_path)
print(f"  ✓ Saved to {final_model_path}")

# Save training history
history = {
    'train_losses': trainer.train_losses,
    'val_losses': trainer.val_losses,
    'best_val_loss': best_val_loss,
    'num_epochs_trained': epoch + 1,
    'config': {
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'model_name': config.model_name,
        'num_conditions': df['condition'].nunique()
    }
}

history_path = config.final_dir / "training_history.json"
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f"  ✓ History saved to {history_path}")

# Plot loss curves
print("\nGenerating loss plot...")
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
epochs = range(1, len(trainer.train_losses) + 1)
plt.plot(epochs, trainer.train_losses, 'b-', label='Train Loss', linewidth=2)
plt.plot(epochs, trainer.val_losses, 'r-', label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress - Labeled Data')
plt.legend()
plt.grid(True, alpha=0.3)

# Improvement plot
plt.subplot(1, 2, 2)
train_improvements = [(trainer.train_losses[0] - loss) for loss in trainer.train_losses]
val_improvements = [(trainer.val_losses[0] - loss) for loss in trainer.val_losses]
plt.plot(epochs, train_improvements, 'b-', label='Train Improvement', linewidth=2)
plt.plot(epochs, val_improvements, 'r-', label='Val Improvement', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss Reduction')
plt.title('Loss Improvement Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = config.final_dir / "training_curves.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Plot saved to {plot_path}")
plt.close()

# Final analysis
print("\n" + "="*80)
print("TRAINING ANALYSIS")
print("="*80)

if len(trainer.train_losses) > 0:
    initial_loss = trainer.train_losses[0]
    final_loss = trainer.train_losses[-1]
    total_improvement = initial_loss - final_loss
    percent_improvement = (total_improvement / initial_loss) * 100

    print(f"Initial train loss: {initial_loss:.4f}")
    print(f"Final train loss: {final_loss:.4f}")
    print(f"Total improvement: {total_improvement:.4f} ({percent_improvement:.1f}%)")

    if percent_improvement > 5:
        print("\n✓ SUCCESS: Model learned significantly!")
        print("  The model can now distinguish between different skin conditions.")
    elif percent_improvement > 1:
        print("\n✓ MODERATE SUCCESS: Model showed some learning.")
        print("  Consider training longer or adjusting hyperparameters.")
    else:
        print("\n⚠ LIMITED LEARNING: Model improvement was minimal.")
        print("  Consider checking data quality or model architecture.")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("1. Evaluate the trained model:")
print("   uv run python evaluate_embedder.py")
print("\n2. Build similarity index with trained embeddings:")
print("   uv run python build_index.py")
print("\n3. Use for inference on new skin images")
print("\nModel saved at:", final_model_path)