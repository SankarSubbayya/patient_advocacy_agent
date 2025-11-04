#!/usr/bin/env python
"""
Fine-tune SigLIP embedder on SCIN dataset.

This script handles:
1. Loading the SCIN dataset
2. Creating SigLIP embedder
3. Training with contrastive loss
4. Saving checkpoints
5. Evaluating performance

Run with: uv run python train_embedder.py
"""

import torch
import logging
from pathlib import Path
from datetime import datetime
import json
import yaml

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


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Config loaded from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load config: {e}, using defaults")
        return {}


class TrainingConfig:
    """Training configuration."""

    def __init__(self, config_path: str = "config.yaml"):
        # Load config from YAML
        config = load_config(config_path)
        
        # Paths - read from config.yaml with fallbacks
        data_config = config.get('data', {})
        model_config = config.get('models', {})
        embedder_config = model_config.get('embedder', {})
        training_config = config.get('training', {})
        embeddings_config = config.get('embeddings', {})
        
        self.data_dir = Path(data_config.get('scin_dir', './data/scin'))
        self.model_dir = Path(embedder_config.get('dir', './models/embedder'))
        self.checkpoint_dir = Path(embedder_config.get('checkpoints_dir', 
                                                        self.model_dir / 'checkpoints'))
        self.final_dir = Path(embedder_config.get('final_dir', 
                                                   self.model_dir / 'final'))

        # Dataset - read from config.yaml with fallbacks
        self.batch_size = training_config.get('batch_size', 32)
        self.num_workers = embeddings_config.get('num_workers', 0)  # Set to 0 for MPS
        self.test_split = data_config.get('test_split', 0.2)
        self.val_split = data_config.get('val_split', 0.1)

        # Model - read from config.yaml with fallbacks
        self.model_name = embedder_config.get('model_name', 'google/siglip-base-patch16-224')
        self.projection_dim = embedder_config.get('projection_dim', 512)
        self.freeze_backbone = embedder_config.get('freeze_backbone', False)

        # Training - read from config.yaml with fallbacks
        self.num_epochs = training_config.get('num_epochs', 20)
        self.learning_rate = training_config.get('learning_rate', 1e-4)
        self.weight_decay = training_config.get('weight_decay', 1e-5)
        self.early_stopping_patience = training_config.get('early_stopping_patience', 3)

        # Device - check config first, then auto-detect
        device_config = config.get('device', {})
        device_type = device_config.get('type', 'auto')
        
        if device_type == 'auto':
            self.device = self._get_device()
        else:
            self.device = device_type
            logger.info(f"Using device from config: {self.device}")

    def _get_device(self) -> str:
        """Get best available device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'data_dir': str(self.data_dir),
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'model_name': self.model_name,
            'projection_dim': self.projection_dim,
            'device': self.device,
        }


def setup_training() -> tuple:
    """Setup training components."""
    config = TrainingConfig()

    print("\n" + "="*80)
    print("Training Configuration (from config.yaml)")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Data dir: {config.data_dir}")
    print(f"Model dir: {config.model_dir}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Model: {config.model_name}")
    print(f"Projection dim: {config.projection_dim}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Test split: {config.test_split}, Val split: {config.val_split}")

    # Create directories
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.final_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n" + "="*80)
    print("Loading Dataset")
    print("="*80)

    if not config.data_dir.exists():
        print(f"✗ Dataset not found at {config.data_dir}")
        print("Run: python download_scin_dataset.py")
        return None, None, None

    data_loader = SCINDataLoader(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        test_split=config.test_split,
        val_split=config.val_split,
    )

    try:
        dataloaders = data_loader.create_dataloaders()
        print(f"✓ Dataset loaded")
        print(f"  - Train: {len(data_loader.train_dataset)} images")
        print(f"  - Val: {len(data_loader.val_dataset)} images")
        print(f"  - Test: {len(data_loader.test_dataset)} images")
        print(f"  - Conditions: {data_loader.get_num_classes()}")
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
    except Exception as e:
        print(f"✗ Failed to create embedder: {e}")
        return None, None, None

    return config, dataloaders, embedder


def train(config: TrainingConfig, dataloaders: dict, embedder) -> bool:
    """Train the embedder."""
    print("\n" + "="*80)
    print("Training Embedder")
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

        # Train
        print(f"\n Starting training...")
        history = trainer.fit(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            num_epochs=config.num_epochs,
            checkpoint_dir=config.checkpoint_dir,
            early_stopping_patience=config.early_stopping_patience,
        )

        print(f"\n✓ Training completed")
        print(f"  - Final train loss: {history['train_losses'][-1]:.4f}")
        print(f"  - Final val loss: {history['val_losses'][-1]:.4f}")

        # Save final model
        print(f"\n Saving final model...")
        embedder.save(config.final_dir / "embedder.pt")
        print(f"✓ Model saved to {config.final_dir / 'embedder.pt'}")

        # Save training history
        history_file = config.final_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✓ Training history saved to {history_file}")

        # Save config
        config_file = config.final_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump({
                **config.to_dict(),
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        print(f"✓ Config saved to {config_file}")

        return True

    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("\n" + "="*80)
    print("SigLIP Embedder Training Script")
    print("="*80)

    # Setup
    config, dataloaders, embedder = setup_training()

    if config is None or dataloaders is None or embedder is None:
        print("\n✗ Failed to setup training")
        return 1

    # Train
    if train(config, dataloaders, embedder):
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
        print("\nNext steps:")
        print("1. Build similarity index:")
        print("   uv run python build_index.py")
        print("\n2. Run assessments:")
        print("   uv run python example_usage.py")
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
