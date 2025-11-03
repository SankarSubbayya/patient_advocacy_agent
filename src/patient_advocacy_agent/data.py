"""Data pipeline for loading and processing skin condition images from SCIN dataset."""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pydantic import BaseModel, Field


class ImageMetadata(BaseModel):
    """Metadata for a skin condition image."""
    image_id: str
    image_path: str
    condition: str
    condition_label: int
    description: Optional[str] = None
    symptoms: Optional[List[str]] = None
    severity: Optional[str] = None
    additional_notes: Optional[str] = None


class SkinConditionDataset(Dataset):
    """PyTorch Dataset for skin condition images with labels."""

    def __init__(
        self,
        image_dir: Path,
        metadata_df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the skin condition dataset.

        Args:
            image_dir: Directory containing skin condition images
            metadata_df: DataFrame with columns: image_id, image_path, condition, condition_label
            transform: Optional torchvision transforms to apply to images
            image_size: Target size for images (H, W)
        """
        self.image_dir = Path(image_dir)
        self.metadata_df = metadata_df
        self.image_size = image_size

        # Default transforms if none provided
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        self.transform = transform

        # Create condition to label mapping
        self.condition_to_label = {
            cond: int(label)
            for cond, label in zip(
                metadata_df['condition'].unique(),
                metadata_df['condition_label'].unique()
            )
        }
        self.label_to_condition = {v: k for k, v in self.condition_to_label.items()}
        self.num_classes = len(self.condition_to_label)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Returns:
            Dictionary containing:
                - image: Tensor of shape (3, H, W)
                - label: Condition label (int)
                - condition: Condition name (str)
                - image_id: Unique identifier (str)
                - metadata: Full metadata dict
        """
        row = self.metadata_df.iloc[idx]
        image_path = self.image_dir / row['image_path']

        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Prepare metadata dict, excluding None values
        metadata_dict = {}
        for k, v in row.items():
            if k in ImageMetadata.__fields__:
                # Skip NaN/None values for optional fields
                if pd.notna(v):
                    metadata_dict[k] = v

        return {
            'image': image,
            'label': int(row['condition_label']),
            'condition': row['condition'],
            'image_id': row['image_id'],
            'metadata': ImageMetadata(**metadata_dict).model_dump()
        }


class SCINDataLoader:
    """Manager for SCIN dataset loading and processing."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        test_split: float = 0.2,
        val_split: float = 0.1
    ):
        """
        Initialize the SCIN data loader.

        Args:
            data_dir: Root directory containing SCIN dataset
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for parallel loading
            test_split: Fraction of data for testing (0.0-1.0)
            val_split: Fraction of data for validation (0.0-1.0)
        """
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".scin_data"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_split = test_split
        self.val_split = val_split

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.condition_labels = None

    def prepare_dataset(
        self,
        metadata_path: Optional[Path] = None,
        images_dir: str = "images"
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Prepare and load the SCIN dataset.

        Args:
            metadata_path: Path to metadata CSV file
            images_dir: Subdirectory containing images relative to data_dir

        Returns:
            Tuple of (metadata_df, condition_label_mapping)
        """
        images_path = self.data_dir / images_dir

        if metadata_path is None:
            metadata_path = self.data_dir / "metadata.csv"

        # Load metadata
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                "Please download SCIN dataset first."
            )

        # Load metadata, ensuring image_id is string (not int)
        df = pd.read_csv(metadata_path, dtype={'image_id': str})

        # Create condition labels if not present
        if 'condition_label' not in df.columns:
            conditions = df['condition'].unique()
            condition_to_label = {cond: idx for idx, cond in enumerate(conditions)}
            df['condition_label'] = df['condition'].map(condition_to_label)
        else:
            condition_to_label = dict(
                zip(df['condition'].unique(), df['condition_label'].unique())
            )

        self.condition_labels = condition_to_label

        # Split data
        n = len(df)
        test_size = int(n * self.test_split)
        val_size = int(n * self.val_split)
        train_size = n - test_size - val_size

        indices = np.random.permutation(n)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }, condition_to_label

    def create_dataloaders(
        self,
        metadata_path: Optional[Path] = None,
        images_dir: str = "images"
    ) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for train/val/test splits.

        Args:
            metadata_path: Path to metadata CSV file
            images_dir: Subdirectory containing images

        Returns:
            Dictionary with keys 'train', 'val', 'test' containing DataLoaders
        """
        splits, condition_labels = self.prepare_dataset(metadata_path, images_dir)
        self.condition_labels = condition_labels

        images_path = self.data_dir / images_dir

        # Create datasets
        self.train_dataset = SkinConditionDataset(
            images_path, splits['train']
        )
        self.val_dataset = SkinConditionDataset(
            images_path, splits['val']
        )
        self.test_dataset = SkinConditionDataset(
            images_path, splits['test']
        )

        # Create dataloaders
        dataloaders = {
            'train': DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            ),
            'val': DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            ),
            'test': DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
        }

        return dataloaders

    def get_condition_labels(self) -> Dict[str, int]:
        """Get mapping of condition names to integer labels."""
        if self.condition_labels is None:
            raise RuntimeError(
                "Dataset not prepared yet. Call prepare_dataset() first."
            )
        return self.condition_labels

    def get_num_classes(self) -> int:
        """Get number of unique conditions."""
        if self.condition_labels is None:
            raise RuntimeError(
                "Dataset not prepared yet. Call prepare_dataset() first."
            )
        return len(self.condition_labels)
