"""SigLIP-based image embedder for fine-tuning on skin condition images."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for pulling similar images together and pushing different ones apart."""

    def __init__(self, temperature: float = 0.07):
        """
        Initialize contrastive loss.

        Args:
            temperature: Temperature parameter for scaling logits
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        logit_scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss (InfoNCE loss).

        Args:
            image_embeddings: Batch of image embeddings (B, D)
            text_embeddings: Batch of text embeddings (B, D)
            logit_scale: Learned temperature scale

        Returns:
            Scalar loss value
        """
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Compute similarity matrix
        logits = logit_scale * (image_embeddings @ text_embeddings.T)

        # Labels: diagonal elements are positive pairs
        batch_size = image_embeddings.shape[0]
        labels = torch.arange(batch_size, device=image_embeddings.device)

        # Contrastive loss: both directions (image->text and text->image)
        loss_img = F.cross_entropy(logits, labels)
        loss_txt = F.cross_entropy(logits.T, labels)
        loss = (loss_img + loss_txt) / 2

        return loss


class SigLIPEmbedder(nn.Module):
    """SigLIP-based image embedder for skin condition classification and retrieval."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        hidden_dim: int = 768,
        projection_dim: int = 512,
        freeze_backbone: bool = False
    ):
        """
        Initialize SigLIP embedder.

        Args:
            model_name: Name of the SigLIP model to use
            hidden_dim: Hidden dimension from the base model
            projection_dim: Dimension for the projection layer
            freeze_backbone: Whether to freeze the backbone model weights
        """
        super().__init__()

        # Load pretrained SigLIP model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Projection head for fine-tuning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Loss function
        self.loss_fn = ContrastiveLoss()

    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features from the model.

        Args:
            images: Batch of images (B, 3, H, W)

        Returns:
            Image embeddings (B, projection_dim)
        """
        # Don't use torch.no_grad() - we need gradients for fine-tuning!
        outputs = self.model.get_image_features(pixel_values=images)

        # Project to lower dimension
        embeddings = self.projection_head(outputs)
        return embeddings

    def extract_text_features(self, texts: list) -> torch.Tensor:
        """
        Extract text features from text inputs.

        Args:
            texts: List of text descriptions

        Returns:
            Text embeddings (B, projection_dim)
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        # Don't use torch.no_grad() - we need gradients for fine-tuning!
        outputs = self.model.get_text_features(**inputs)

        # Project to lower dimension
        embeddings = self.projection_head(outputs)
        return embeddings

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for image embeddings.

        Args:
            images: Batch of images (B, 3, H, W)

        Returns:
            Image embeddings (B, projection_dim)
        """
        return self.extract_image_features(images)

    def compute_loss(
        self,
        images: torch.Tensor,
        text_descriptions: list
    ) -> torch.Tensor:
        """
        Compute contrastive loss between images and text.

        Args:
            images: Batch of images (B, 3, H, W)
            text_descriptions: List of text descriptions (B,)

        Returns:
            Scalar loss value
        """
        # Move to same device as model
        device = next(self.parameters()).device
        images = images.to(device)

        # Extract embeddings
        image_embeddings = self.extract_image_features(images)
        text_embeddings = self.extract_text_features(text_descriptions)

        # Compute contrastive loss
        loss = self.loss_fn(
            image_embeddings,
            text_embeddings,
            torch.exp(self.logit_scale)
        )

        return loss

    def save(self, path: Path) -> None:
        """Save embedder checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'logit_scale': self.logit_scale.item()
        }, path)
        logger.info(f"Embedder saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "SigLIPEmbedder":
        """Load embedder from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        embedder = cls(model_name=checkpoint['model_name'])
        embedder.load_state_dict(checkpoint['model_state_dict'])
        return embedder


class EmbedderTrainer:
    """Trainer for fine-tuning the SigLIP embedder."""

    def __init__(
        self,
        embedder: SigLIPEmbedder,
        device: Optional[str] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer.

        Args:
            embedder: SigLIPEmbedder instance
            device: Device to use ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.embedder = embedder
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedder.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.embedder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = None

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader with batches of (images, text_descriptions, labels)

        Returns:
            Average loss for the epoch
        """
        self.embedder.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch['image'].to(self.device)
            text_descriptions = batch.get('description', [str(c) for c in batch['condition']])

            self.optimizer.zero_grad()

            loss = self.embedder.compute_loss(images, text_descriptions)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self, val_loader) -> float:
        """
        Validate on validation set.

        Args:
            val_loader: DataLoader with validation batches

        Returns:
            Average validation loss
        """
        self.embedder.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                text_descriptions = batch.get('description', [str(c) for c in batch['condition']])

                loss = self.embedder.compute_loss(images, text_descriptions)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)

        return avg_loss

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 10,
        checkpoint_dir: Optional[Path] = None,
        early_stopping_patience: int = 3
    ) -> Dict[str, list]:
        """
        Train the embedder.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping

        Returns:
            Dictionary with training history
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                if checkpoint_dir:
                    checkpoint_path = Path(checkpoint_dir) / f"embedder_epoch_{epoch + 1}.pt"
                    self.embedder.save(checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if self.scheduler:
                self.scheduler.step()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
