#!/usr/bin/env python
"""
Fine-tune SigLIP embedder on SCIN dataset with LABELED data.

This script uses the synthetic_labeled_metadata.csv which contains:
- Multiple skin condition classes
- Proper text descriptions for contrastive learning
- Train/val/test splits

Run with: uv run python train_embedder_labeled.py
"""

import torch
import logging
from pathlib import Path
from datetime import datetime
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from patient_advocacy_agent import (
    SCINDataLoader,
    SigLIPEmbedder,
    EmbedderTrainer,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LabeledTrainingConfig:
    """Training configuration for labeled data."""

    def __init__(self):
        # Paths - use LABELED metadata
        self.data_dir = Path('/home/sankar/data/scin')
        self.metadata_file = 'synthetic_labeled_metadata.csv'  # USE LABELED DATA
        self.images_dir = 'images'  # Original images directory

        self.model_dir = Path('/home/sankar/models/embedder_labeled')
        self.checkpoint_dir = self.model_dir / 'checkpoints'
        self.final_dir = self.model_dir / 'final'

        # Dataset
        self.batch_size = 32
        self.num_workers = 0  # Set to 0 for MPS/debugging
        self.test_split = 0.15  # Already split in metadata
        self.val_split = 0.15

        # Model
        self.model_name = 'google/siglip-base-patch16-224'
        self.projection_dim = 512
        self.freeze_backbone = False  # Fine-tune entire model

        # Training
        self.num_epochs = 20
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.early_stopping_patience = 5

        # Device
        self.device = self._get_device()

    def _get_device(self) -> str:
        """Get best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


def verify_labeled_data():
    """Verify that we have properly labeled data."""
    config = LabeledTrainingConfig()
    metadata_path = config.data_dir / config.metadata_file

    print("\n" + "="*80)
    print("Verifying Labeled Data")
    print("="*80)

    df = pd.read_csv(metadata_path)

    print(f"\nMetadata file: {metadata_path}")
    print(f"Total samples: {len(df)}")

    # Check conditions
    conditions = df['condition'].value_counts()
    print(f"\nUnique conditions: {len(conditions)}")
    print("\nTop 10 conditions:")
    for condition, count in conditions.head(10).items():
        print(f"  {condition}: {count}")

    # Check splits
    splits = df['split'].value_counts()
    print("\nData splits:")
    for split, count in splits.items():
        print(f"  {split}: {count}")

    # Check for descriptions
    if 'description' in df.columns:
        print(f"\nHas text descriptions: Yes")
        print("Sample descriptions:")
        for desc in df['description'].sample(3).values:
            print(f"  - {desc}")
    else:
        print(f"\nHas text descriptions: No")

    return len(conditions) > 1  # Return True if we have multiple classes


def setup_training():
    """Setup training components with labeled data."""
    config = LabeledTrainingConfig()

    print("\n" + "="*80)
    print("Training Configuration (LABELED DATA)")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Data dir: {config.data_dir}")
    print(f"Metadata: {config.metadata_file} (LABELED)")
    print(f"Model dir: {config.model_dir}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")

    # Create directories
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.final_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset with labeled metadata
    print("\n" + "="*80)
    print("Loading LABELED Dataset")
    print("="*80)

    metadata_path = config.data_dir / config.metadata_file

    if not metadata_path.exists():
        print(f"✗ Labeled metadata not found at {metadata_path}")
        print("Run: python create_synthetic_labels.py")
        return None, None, None

    # Custom data loader that uses the labeled metadata
    data_loader = SCINDataLoader(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        test_split=config.test_split,
        val_split=config.val_split,
    )

    try:
        # Pass the labeled metadata path explicitly
        dataloaders = data_loader.create_dataloaders(
            metadata_path=metadata_path,
            images_dir=config.images_dir
        )

        print(f"✓ Dataset loaded with LABELED data")
        print(f"  - Train: {len(data_loader.train_dataset)} images")
        print(f"  - Val: {len(data_loader.val_dataset)} images")
        print(f"  - Test: {len(data_loader.test_dataset)} images")
        print(f"  - Conditions: {data_loader.get_num_classes()} (MULTIPLE CLASSES!)")

    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return None, None, None

    # Create embedder
    print("\n" + "="*80)
    print("Creating Embedder")
    print("="*80)

    try:
        embedder = SigLIPEmbedder(
            model_name=config.model_name,
            projection_dim=config.projection_dim,
            freeze_backbone=config.freeze_backbone,
        )
        print(f"✓ Embedder created")
        print(f"  - Model: {config.model_name}")
        print(f"  - Projection dim: {config.projection_dim}")
        print(f"  - Freeze backbone: {config.freeze_backbone}")

    except Exception as e:
        print(f"✗ Failed to create embedder: {e}")
        return None, None, None

    return config, dataloaders, embedder


def train_with_monitoring(config, dataloaders, embedder):
    """Train the embedder with real-time loss monitoring."""
    print("\n" + "="*80)
    print("Training Embedder with LABELED Data")
    print("="*80)

    try:
        # Create trainer
        trainer = EmbedderTrainer(
            embedder=embedder,
            device=config.device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        print(f"✓ Trainer created")
        print(f"  - Device: {config.device}")
        print(f"  - Learning rate: {config.learning_rate}")

        # Train with monitoring
        print(f"\nStarting training with LABELED data...")
        print("This should show DECREASING loss (not flat!)")
        print("-" * 80)

        history = trainer.fit(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            num_epochs=config.num_epochs,
            checkpoint_dir=config.checkpoint_dir,
            early_stopping_patience=config.early_stopping_patience,
        )

        print(f"\n✓ Training completed")

        # Analyze training results
        train_losses = history['train_losses']
        val_losses = history['val_losses']

        print("\nTraining Analysis:")
        print(f"  - Initial train loss: {train_losses[0]:.4f}")
        print(f"  - Final train loss: {train_losses[-1]:.4f}")
        print(f"  - Loss reduction: {(train_losses[0] - train_losses[-1]):.4f}")

        if train_losses[0] - train_losses[-1] > 0.01:
            print("  ✓ Model is LEARNING! Loss decreased significantly")
        else:
            print("  ⚠ Loss barely changed - check learning rate or data")

        # Save final model
        print(f"\nSaving final model...")
        embedder.save(config.final_dir / "embedder_labeled.pt")
        print(f"✓ Model saved to {config.final_dir / 'embedder_labeled.pt'}")

        # Save training history
        history_file = config.final_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✓ Training history saved to {history_file}")

        # Plot loss curves
        plot_losses(history, config.final_dir)

        return True

    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def plot_losses(history, output_dir):
    """Plot and save loss curves."""
    try:
        plt.figure(figsize=(10, 6))

        epochs = range(1, len(history['train_losses']) + 1)
        plt.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
        plt.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('SigLIP Fine-tuning with LABELED Data')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add annotations
        initial_loss = history['train_losses'][0]
        final_loss = history['train_losses'][-1]
        plt.annotate(f'Start: {initial_loss:.4f}',
                    xy=(1, initial_loss),
                    xytext=(2, initial_loss + 0.1))
        plt.annotate(f'End: {final_loss:.4f}',
                    xy=(len(epochs), final_loss),
                    xytext=(len(epochs)-2, final_loss + 0.1))

        plot_path = output_dir / 'loss_curves_labeled.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Loss plot saved to {plot_path}")
        plt.close()

    except Exception as e:
        print(f"Warning: Could not create plot: {e}")


def main():
    """Main training function."""
    print("\n" + "="*80)
    print("SigLIP Embedder Training with LABELED Data")
    print("="*80)

    # First verify we have proper labeled data
    if not verify_labeled_data():
        print("\n✗ No labeled data found or only single class!")
        print("Run: python create_synthetic_labels.py")
        return 1

    # Setup
    config, dataloaders, embedder = setup_training()

    if config is None or dataloaders is None or embedder is None:
        print("\n✗ Failed to setup training")
        return 1

    # Train
    if train_with_monitoring(config, dataloaders, embedder):
        print("\n" + "="*80)
        print("Training Complete with LABELED Data!")
        print("="*80)
        print("\nThe model should now have learned to distinguish between")
        print("different skin conditions using contrastive learning.")
        print("\nNext steps:")
        print("1. Evaluate the model:")
        print("   uv run python evaluate_embedder.py")
        print("\n2. Build similarity index:")
        print("   uv run python build_index.py")
        return 0
    else:
        return 1


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)