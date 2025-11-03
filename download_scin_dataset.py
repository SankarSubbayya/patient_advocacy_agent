#!/usr/bin/env python
"""
Download and prepare SCIN (Skin Condition Image Network) dataset.

This script helps you download the SCIN dataset from GitHub and organize it
for use with the patient advocacy agent.

SCIN Dataset: https://github.com/ISMAE-SUDA/SCIN
"""

import os
import sys
import shutil
import urllib.request
import zipfile
from pathlib import Path
import json
from typing import Dict, List, Optional
import csv


class SCINDownloader:
    """Handle SCIN dataset downloading and preparation."""

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize downloader.

        Args:
            base_dir: Base directory for dataset (default: ./data/)
        """
        self.base_dir = Path(base_dir) if base_dir else Path("./data")
        self.dataset_dir = self.base_dir / "scin"
        self.images_dir = self.dataset_dir / "images"
        self.metadata_file = self.dataset_dir / "metadata.csv"

    def check_existing_dataset(self) -> bool:
        """Check if dataset already exists."""
        if self.dataset_dir.exists() and self.images_dir.exists():
            # Check for images in current directory or subdirectories
            image_count = len(list(self.images_dir.glob("*.jpg"))) + \
                         len(list(self.images_dir.glob("*.png"))) + \
                         len(list(self.images_dir.glob("*/*.jpg"))) + \
                         len(list(self.images_dir.glob("*/*.png")))
            if image_count > 0:
                print(f"✓ Found existing dataset with {image_count} images")
                return True
        return False

    def create_directories(self) -> None:
        """Create necessary directories."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directories: {self.dataset_dir}")

    def download_dataset(self) -> bool:
        """
        Download SCIN dataset from GitHub.

        Note: This requires manual download from:
        https://github.com/ISMAE-SUDA/SCIN/releases

        This function provides instructions for downloading.
        """
        print("\n" + "="*80)
        print("SCIN Dataset Download Instructions")
        print("="*80)

        print("""
The SCIN dataset is available on GitHub but requires registration/download.

Follow these steps:

1. Visit: https://github.com/ISMAE-SUDA/SCIN
2. Look for the "Releases" section
3. Download the dataset ZIP file (usually SCIN-main.zip or similar)
4. Extract to this location: {dataset_dir}

OR use this command (if wget available):
    wget -O scin.zip https://github.com/ISMAE-SUDA/SCIN/archive/refs/heads/main.zip
    unzip scin.zip -d {dataset_dir}
    # Then organize the files

Alternative datasets you can use:

A. Skin Lesion Analysis Toward Melanoma Detection (ISIC):
   https://www.isic-archive.com/
   Format: JPEG images with JSON metadata

B. DermNet:
   https://www.dermnetnz.org/
   Format: Various formats, see documentation

C. Fitzpatrick 17k:
   https://github.com/mattgroff/fitzpatrick17k
   Format: Organized by condition

D. Create your own dataset:
   See DATASET_GUIDE.md for creating custom datasets
""".format(dataset_dir=self.dataset_dir))

        return False

    def organize_dataset(self, source_dir: Optional[Path] = None) -> bool:
        """
        Organize downloaded dataset.

        Args:
            source_dir: Source directory containing images

        Returns:
            True if organization successful
        """
        if source_dir is None:
            source_dir = self.dataset_dir

        if not source_dir.exists():
            print(f"✗ Source directory not found: {source_dir}")
            return False

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = []

        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(Path(root) / file)

        if not image_files:
            print(f"✗ No image files found in {source_dir}")
            return False

        print(f"\n✓ Found {len(image_files)} images")

        # Copy images to images directory
        print(f"✓ Copying images to {self.images_dir}...")
        for i, src_file in enumerate(image_files, 1):
            dst_file = self.images_dir / src_file.name

            # Avoid name conflicts
            if dst_file.exists():
                name_parts = src_file.stem.split('_')
                dst_file = self.images_dir / f"{src_file.stem}_{i}{src_file.suffix}"

            try:
                shutil.copy2(src_file, dst_file)
                if i % 100 == 0:
                    print(f"  Copied {i}/{len(image_files)} images...")
            except Exception as e:
                print(f"  Warning: Could not copy {src_file}: {e}")

        print(f"✓ Copied {len(image_files)} images")
        return True

    def create_metadata_csv(self, conditions: Optional[Dict[str, int]] = None) -> bool:
        """
        Create metadata CSV file from image directory.

        Args:
            conditions: Optional mapping of condition names to labels

        Returns:
            True if metadata created successfully
        """
        if not self.images_dir.exists():
            print(f"✗ Images directory not found: {self.images_dir}")
            return False

        # Get all image files
        image_files = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.jpeg")) +
            list(self.images_dir.glob("*.png"))
        )

        if not image_files:
            print(f"✗ No images found in {self.images_dir}")
            return False

        print(f"\n✓ Found {len(image_files)} images")
        print(f"✓ Creating metadata file...")

        # Default conditions (can be customized)
        if conditions is None:
            conditions = {
                'unknown': 0,  # For images without clear label
                'eczema': 1,
                'psoriasis': 2,
                'acne': 3,
                'dermatitis': 4,
                'rosacea': 5,
                'urticaria': 6,
                'fungal_infection': 7,
                'viral_infection': 8,
                'bacterial_infection': 9,
                'other': 10,
            }

        # Create metadata
        metadata = []
        for i, image_path in enumerate(image_files):
            # Try to infer condition from filename
            filename_lower = image_path.stem.lower()
            condition = 'unknown'

            for cond_name in conditions.keys():
                if cond_name.lower() in filename_lower:
                    condition = cond_name
                    break

            entry = {
                'image_id': f'case_{i:06d}',
                'image_path': image_path.name,
                'condition': condition,
                'condition_label': conditions.get(condition, 0),
                'symptoms': '',
                'severity': 'unknown',
                'notes': '',
            }
            metadata.append(entry)

        # Write CSV
        try:
            with open(self.metadata_file, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=['image_id', 'image_path', 'condition',
                               'condition_label', 'symptoms', 'severity', 'notes']
                )
                writer.writeheader()
                writer.writerows(metadata)

            print(f"✓ Created metadata file: {self.metadata_file}")
            print(f"  - Total images: {len(metadata)}")
            print(f"  - Conditions: {len(set(m['condition'] for m in metadata))}")
            return True

        except Exception as e:
            print(f"✗ Failed to create metadata: {e}")
            return False

    def validate_dataset(self) -> bool:
        """Validate dataset structure."""
        print("\n" + "="*80)
        print("Dataset Validation")
        print("="*80)

        # Check directories
        if not self.dataset_dir.exists():
            print("✗ Dataset directory not found")
            return False
        print(f"✓ Dataset directory: {self.dataset_dir}")

        if not self.images_dir.exists():
            print("✗ Images directory not found")
            return False

        # Count images (in current directory or subdirectories)
        images = list(self.images_dir.glob("*.jpg")) + \
                 list(self.images_dir.glob("*.png")) + \
                 list(self.images_dir.glob("*.jpeg")) + \
                 list(self.images_dir.glob("*/*.jpg")) + \
                 list(self.images_dir.glob("*/*.png")) + \
                 list(self.images_dir.glob("*/*.jpeg"))
        print(f"✓ Images found: {len(images)}")

        if len(images) == 0:
            print("✗ No images found")
            return False

        # Check metadata
        if not self.metadata_file.exists():
            print("✗ Metadata file not found")
            return False
        print(f"✓ Metadata file: {self.metadata_file}")

        # Validate metadata
        try:
            with open(self.metadata_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            print(f"✓ Metadata entries: {len(rows)}")

            if len(rows) != len(images):
                print(f"⚠ Warning: {len(images)} images but {len(rows)} metadata entries")

            # Check conditions distribution
            conditions = {}
            for row in rows:
                cond = row['condition']
                conditions[cond] = conditions.get(cond, 0) + 1

            print(f"✓ Condition distribution:")
            for cond, count in sorted(conditions.items()):
                print(f"  - {cond:20} {count:5} ({count*100//len(rows)}%)")

        except Exception as e:
            print(f"✗ Failed to validate metadata: {e}")
            return False

        print("\n✓ Dataset validation passed!")
        return True

    def get_dataset_info(self) -> Dict:
        """Get information about the dataset."""
        info = {
            'dataset_dir': str(self.dataset_dir),
            'images_dir': str(self.images_dir),
            'metadata_file': str(self.metadata_file),
            'images_count': 0,
            'conditions': {},
        }

        if self.images_dir.exists():
            info['images_count'] = len(
                list(self.images_dir.glob("*.jpg")) +
                list(self.images_dir.glob("*.png")) +
                list(self.images_dir.glob("*.jpeg")) +
                list(self.images_dir.glob("*/*.jpg")) +
                list(self.images_dir.glob("*/*.png")) +
                list(self.images_dir.glob("*/*.jpeg"))
            )

        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cond = row['condition']
                    info['conditions'][cond] = info['conditions'].get(cond, 0) + 1

        return info


def main():
    """Main function."""
    print("\n" + "="*80)
    print("SCIN Dataset Download and Preparation Tool")
    print("="*80)

    # Create downloader
    downloader = SCINDownloader(base_dir=Path("./data"))

    # Check if dataset already exists
    print("\n✓ Checking for existing dataset locally...")
    if downloader.check_existing_dataset():
        print("\n✓ ✓ ✓ Dataset already exists locally! ✓ ✓ ✓")
        print("\nYou don't need to download. Using existing data...")

        # Validate existing dataset
        if downloader.validate_dataset():
            info = downloader.get_dataset_info()
            print(f"\n✓ Dataset Ready to Use:")
            print(f"  Location: {info['dataset_dir']}")
            print(f"  Images: {info['images_count']}")
            print(f"  Conditions: {len(info['conditions'])}")
            print(f"\n✓ Next step: Run training")
            print(f"  Command: uv run python train_embedder.py")
            return
        else:
            print("\n⚠ Existing dataset validation failed. Proceeding to re-organize...")
    else:
        print("✗ No existing dataset found.")

    # Create directories
    downloader.create_directories()

    # Download instructions
    print("\n" + "="*80)
    print("Step 1: Download Dataset")
    print("="*80)

    downloader.download_dataset()

    # Ask if user has dataset ready
    response = input("\nDo you have the dataset ready? (y/n): ").strip().lower()
    if response != 'y':
        print("\nPlease download the dataset first, then run this script again.")
        return

    # Organize dataset
    print("\n" + "="*80)
    print("Step 2: Organize Dataset")
    print("="*80)

    source_dir_input = input(
        f"Enter source directory (press Enter for {downloader.dataset_dir}): "
    ).strip()

    if source_dir_input:
        source_dir = Path(source_dir_input)
    else:
        source_dir = downloader.dataset_dir

    if downloader.organize_dataset(source_dir):
        print("✓ Dataset organization completed")
    else:
        print("✗ Failed to organize dataset")
        return

    # Create metadata
    print("\n" + "="*80)
    print("Step 3: Create Metadata")
    print("="*80)

    if downloader.create_metadata_csv():
        print("✓ Metadata creation completed")
    else:
        print("✗ Failed to create metadata")
        return

    # Validate
    print("\n" + "="*80)
    print("Step 4: Validate Dataset")
    print("="*80)

    if downloader.validate_dataset():
        print("\n✓ Dataset is ready to use!")

        # Show info
        info = downloader.get_dataset_info()
        print("\nDataset Information:")
        print(f"  Location: {info['dataset_dir']}")
        print(f"  Images: {info['images_count']}")
        print(f"  Conditions: {len(info['conditions'])}")

        print("\nNext steps:")
        print("1. Review metadata: cat data/scin/metadata.csv")
        print("2. Fine-tune model: uv run python train_embedder.py")
        print("3. Build index: uv run python build_index.py")
    else:
        print("\n✗ Dataset validation failed")


if __name__ == "__main__":
    import sys

    try:
        # Check for non-interactive mode (e.g., CI/CD)
        if not sys.stdin.isatty():
            print("\n" + "="*80)
            print("Running in non-interactive mode")
            print("="*80)

            downloader = SCINDownloader()

            # Check for existing data
            print("\n✓ Checking for existing dataset locally...")
            if downloader.check_existing_dataset():
                print("\n✓ ✓ ✓ Dataset already exists locally! ✓ ✓ ✓")
                print("\nUsing existing data at:", downloader.dataset_dir)

                # Just validate and show info
                if downloader.validate_dataset():
                    info = downloader.get_dataset_info()
                    print("\nDataset Information:")
                    print(f"  Location: {info['dataset_dir']}")
                    print(f"  Images: {info['images_count']}")
                    print(f"  Conditions: {len(info['conditions'])}")
                    print("\n✓ Dataset is ready for training!")
                else:
                    print("✗ Validation failed")
                    sys.exit(1)
            else:
                print("\n✗ No existing dataset found and running in non-interactive mode.")
                print("\nTo download the dataset, please:")
                print("1. Visit: https://github.com/ISMAE-SUDA/SCIN")
                print("2. Download the dataset")
                print("3. Run this script in interactive mode")
                print("\nOr manually place images in: data/scin/images/")
                sys.exit(1)
        else:
            main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
