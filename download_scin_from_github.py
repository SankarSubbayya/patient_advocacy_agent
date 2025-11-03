#!/usr/bin/env python3
"""
Download SCIN dataset from GitHub repository using Python.
Repository: https://github.com/ISMAE-SUDA/SCIN
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path
import json

def download_file(url, filename, chunk_size=8192):
    """Download a file with progress indication."""
    print(f"Downloading: {url}")
    print(f"Saving to: {filename}")
    
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filename, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"  Progress: {percent:.1f}% ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)", end='\r')
        
        print(f"\n✓ Download complete: {filename}")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract ZIP file."""
    print(f"\nExtracting {zip_path} to {extract_to}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extraction complete")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

def organize_images(source_dir, target_dir):
    """Organize downloaded images into the expected structure."""
    print(f"\nOrganizing images from {source_dir} to {target_dir}...")
    
    try:
        # Create target directory
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        # Find and copy all image files
        image_count = 0
        
        # Search for images in the extracted directory
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_path = os.path.join(root, file)
                    
                    # Determine target location
                    # Try to keep condition folder structure
                    rel_path = os.path.relpath(src_path, source_dir)
                    target_path = os.path.join(target_dir, rel_path)
                    
                    # Create subdirectory if needed
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(src_path, target_path)
                    image_count += 1
                    
                    if image_count % 100 == 0:
                        print(f"  Copied {image_count} images...", end='\r')
        
        print(f"\n✓ Organized {image_count} images")
        return image_count
    except Exception as e:
        print(f"✗ Organization failed: {e}")
        return 0

def create_metadata_csv(images_dir, output_csv):
    """Create metadata CSV from image files."""
    print(f"\nCreating metadata CSV...")
    
    try:
        import csv
        
        metadata = []
        
        # Walk through all image files
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Get condition from folder name
                    condition = os.path.basename(root) if root != images_dir else 'unknown'
                    
                    # Get relative path
                    rel_path = os.path.relpath(os.path.join(root, file), images_dir)
                    
                    metadata.append({
                        'image_path': rel_path,
                        'condition': condition,
                        'split': 'train'  # Default to train split
                    })
        
        # Write CSV
        if metadata:
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['image_path', 'condition', 'split'])
                writer.writeheader()
                writer.writerows(metadata)
            
            print(f"✓ Created metadata CSV with {len(metadata)} entries")
            return True
        else:
            print("✗ No images found to create metadata")
            return False
    except Exception as e:
        print(f"✗ Metadata creation failed: {e}")
        return False

def main():
    """Main download function."""
    print("="*80)
    print("SCIN Dataset Downloader from GitHub")
    print("="*80)
    print()
    
    # Configuration
    github_url = "https://github.com/ISMAE-SUDA/SCIN/archive/refs/heads/main.zip"
    zip_filename = "scin_dataset.zip"
    extract_dir = "scin_extracted"
    data_dir = "data/scin"
    images_dir = os.path.join(data_dir, "images")
    metadata_csv = os.path.join(data_dir, "metadata.csv")
    
    print(f"Repository: https://github.com/ISMAE-SUDA/SCIN")
    print(f"Download URL: {github_url}")
    print(f"Target directory: {data_dir}")
    print()
    
    # Step 1: Download
    print("Step 1: Downloading from GitHub")
    print("-" * 80)
    if not download_file(github_url, zip_filename):
        print("Failed to download. Exiting.")
        sys.exit(1)
    
    # Step 2: Extract
    print("\nStep 2: Extracting ZIP file")
    print("-" * 80)
    if not extract_zip(zip_filename, extract_dir):
        print("Failed to extract. Exiting.")
        sys.exit(1)
    
    # Step 3: Organize
    print("\nStep 3: Organizing images")
    print("-" * 80)
    image_count = organize_images(extract_dir, images_dir)
    if image_count == 0:
        print("No images found. Exiting.")
        sys.exit(1)
    
    # Step 4: Create metadata
    print("\nStep 4: Creating metadata CSV")
    print("-" * 80)
    if not create_metadata_csv(images_dir, metadata_csv):
        print("Failed to create metadata. Exiting.")
        sys.exit(1)
    
    # Step 5: Cleanup
    print("\nStep 5: Cleaning up temporary files")
    print("-" * 80)
    try:
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
            print(f"✓ Removed {zip_filename}")
        
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
            print(f"✓ Removed {extract_dir}")
    except Exception as e:
        print(f"⚠ Cleanup warning: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("✓ DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"\nDataset Information:")
    print(f"  Location: {data_dir}")
    print(f"  Images: {image_count}")
    print(f"  Metadata: {metadata_csv}")
    print(f"\nNext steps:")
    print(f"  1. Verify the dataset:")
    print(f"     uv run python verify_setup.py")
    print(f"\n  2. Train the embedder:")
    print(f"     uv run python train_embedder.py")
    print(f"\n  3. Build the index:")
    print(f"     uv run python build_index.py")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
