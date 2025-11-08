#!/usr/bin/env python3
"""
Clean up and organize the patient advocacy agent project.
This script will:
1. Create organized directory structure
2. Move files to appropriate directories
3. Archive old logs
4. Remove temporary files
5. Create a project summary
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime

# Define the new directory structure
DIRECTORIES = {
    'experiments': 'Training scripts and experiments',
    'experiments/contrastive': 'Contrastive learning experiments',
    'experiments/hierarchical': 'Hierarchical model experiments',
    'experiments/fine_grained': 'Fine-grained model experiments',
    'experiments/analysis': 'Analysis and visualization scripts',
    'logs': 'Training and experiment logs',
    'plots': 'Generated visualizations and plots',
    'utils': 'Utility scripts',
    'archive': 'Archived/old files',
}

# File organization mapping
FILE_MAPPING = {
    'experiments/contrastive': [
        'train_siglip_contrastive.py',
        'train_siglip_weighted.py',
    ],
    'experiments/hierarchical': [
        'train_siglip_hierarchical.py',
        'evaluate_hierarchical_retrieval.py',
    ],
    'experiments/fine_grained': [
        'train_siglip_fine_grained.py',
    ],
    'experiments/analysis': [
        'analyze_conditions.py',
        'cluster_embeddings.py',
        'compare_embeddings.py',
        'plot_training_losses.py',
        'plot_real_label_training.py',
        'plot_fine_grained_loss.py',
        'plot_loss_text.py',
    ],
    'utils': [
        'create_coarse_metadata.py',
        'create_correct_metadata.py',
        'create_synthetic_labels.py',
        'download_labeled_images.py',
        'download_correct_labeled_images.py',
        'download_scin_labels.py',
        'merge_scin_labels.py',
        'check_scin_labels.py',
        'test_labeled_data.py',
        'test_gcs_access.py',
        'check_device.py',
        'build_index.py',
    ],
    'logs': [
        '*.log',
    ],
    'plots': [
        '*.png',
        '*.json',  # visualization results
    ],
    'archive': [
        'train_embedder_coarse.py',
        'train_embedder_labeled.py',
        'train_full_labeled.py',
        'train_siglip_hf.py',
        'train_simple.py',
        'download_scin_gcs_working.py',
        'monitor_training.sh',
        'training_log_hf.txt',
    ],
}

# Files to keep in root
KEEP_IN_ROOT = [
    'claude_integration_example.py',
    'config.yaml',
    'README.md',
    'CLAUDE_SETUP.md',
    'DATA_SHARING.md',
    'DATASET_DOWNLOAD_OPTIONS.md',
    'COARSE_CATEGORIES_README.md',
    'build_docs.sh',
    'requirements.txt',
    'pyproject.toml',
    'setup.py',
    '.gitignore',
    'cleanup_project.py',  # This script itself
]

def create_directories():
    """Create the organized directory structure."""
    print("Creating directory structure...")
    for dir_path, description in DIRECTORIES.items():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # Create README in each directory
        readme_path = Path(dir_path) / 'README.md'
        if not readme_path.exists():
            with open(readme_path, 'w') as f:
                f.write(f"# {dir_path.replace('/', ' > ')}\n\n{description}\n")
    print(f"✓ Created {len(DIRECTORIES)} directories")

def move_files():
    """Move files to their appropriate directories."""
    moved_count = 0
    errors = []

    for target_dir, patterns in FILE_MAPPING.items():
        Path(target_dir).mkdir(parents=True, exist_ok=True)

        for pattern in patterns:
            if '*' in pattern:
                # Handle wildcard patterns
                import glob
                files = glob.glob(pattern)
                for file in files:
                    if os.path.basename(file) not in KEEP_IN_ROOT:
                        try:
                            src = Path(file)
                            if src.exists() and src.is_file():
                                dst = Path(target_dir) / src.name
                                if not dst.exists():
                                    shutil.move(str(src), str(dst))
                                    print(f"  Moved {src.name} → {target_dir}/")
                                    moved_count += 1
                        except Exception as e:
                            errors.append(f"Error moving {file}: {e}")
            else:
                # Handle specific files
                src = Path(pattern)
                if src.exists() and src.is_file():
                    dst = Path(target_dir) / src.name
                    if not dst.exists():
                        try:
                            shutil.move(str(src), str(dst))
                            print(f"  Moved {src.name} → {target_dir}/")
                            moved_count += 1
                        except Exception as e:
                            errors.append(f"Error moving {pattern}: {e}")

    print(f"\n✓ Moved {moved_count} files")
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")

    return moved_count

def create_project_summary():
    """Create a summary of the project structure."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'directories': {},
        'models': {},
        'total_files': 0,
        'total_size_mb': 0
    }

    # Count files in each directory
    for dir_path in DIRECTORIES.keys():
        path = Path(dir_path)
        if path.exists():
            files = list(path.glob('*'))
            file_count = len([f for f in files if f.is_file()])
            summary['directories'][dir_path] = file_count
            summary['total_files'] += file_count

    # Check for models
    model_dirs = Path('/home/sankar/models').glob('siglip_*') if Path('/home/sankar/models').exists() else []
    for model_dir in model_dirs:
        if model_dir.is_dir():
            summary_file = model_dir / 'training_summary.json'
            if summary_file.exists():
                with open(summary_file) as f:
                    model_data = json.load(f)
                    summary['models'][model_dir.name] = {
                        'approach': model_data.get('approach', 'unknown'),
                        'accuracy': model_data.get('retrieval_accuracy', 0)
                    }

    # Calculate total size
    import subprocess
    result = subprocess.run(['du', '-sm', '.'], capture_output=True, text=True)
    if result.returncode == 0:
        size_mb = int(result.stdout.split()[0])
        summary['total_size_mb'] = size_mb

    # Write summary
    with open('PROJECT_SUMMARY.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n✓ Created PROJECT_SUMMARY.json")
    return summary

def create_main_readme():
    """Create or update the main README with project structure."""
    readme_content = """# Patient Advocacy Agent - Vision-Language Model

## Project Overview
This project implements and evaluates different approaches for training SigLIP vision-language models on the SCIN dermatology dataset for skin condition classification.

## Project Structure

```
patient_advocacy_agent/
├── experiments/          # Training scripts and experiments
│   ├── contrastive/     # Basic contrastive learning
│   ├── hierarchical/    # Hierarchical (coarse + fine) approach
│   ├── fine_grained/    # Fine-grained 66-class approach
│   └── analysis/        # Analysis and visualization scripts
├── logs/                # Training and experiment logs
├── plots/               # Generated visualizations
├── utils/               # Utility scripts for data processing
├── archive/             # Old/deprecated scripts
├── src/                 # Main source code
│   └── patient_advocacy_agent/
├── docs/                # Documentation
└── models/              # Saved models (external directory)
```

## Model Performance Summary

| Model Approach | Classes | Retrieval Accuracy | Status |
|---------------|---------|-------------------|--------|
| Fine-Grained Contrastive | 66 | 20% | ✅ Best |
| Basic Contrastive | 16 | 20% | ✅ Best |
| Hierarchical | 16+66 | 13% | ⚠️ Lower |

## Quick Start

1. **Training a model:**
   ```bash
   python experiments/fine_grained/train_siglip_fine_grained.py
   ```

2. **Evaluating a model:**
   ```bash
   python experiments/hierarchical/evaluate_hierarchical_retrieval.py
   ```

3. **Using the trained model:**
   ```python
   from patient_advocacy_agent import VisionLanguageEmbedder

   embedder = VisionLanguageEmbedder(
       model_path="/home/sankar/models/siglip_fine_grained/final_model"
   )
   ```

## Key Files

- `claude_integration_example.py` - Example integration with Claude
- `config.yaml` - Main configuration file
- `DATA_SHARING.md` - Data sharing and privacy guidelines
- `COARSE_CATEGORIES_README.md` - Description of the 16 coarse categories

## Requirements

See `requirements.txt` for dependencies.

## Recent Updates

- Trained and evaluated three different SigLIP model approaches
- Fine-grained model achieves 20% retrieval accuracy (2x random baseline)
- Project reorganized for better structure and maintainability
"""

    with open('README.md', 'w') as f:
        f.write(readme_content)

    print("✓ Updated README.md")

def cleanup_temp_files():
    """Remove temporary and unnecessary files."""
    temp_patterns = [
        '__pycache__',
        '*.pyc',
        '.DS_Store',
        'Thumbs.db',
        '*~',
        '*.swp',
    ]

    removed_count = 0
    for pattern in temp_patterns:
        import glob
        for item in glob.glob(f'**/{pattern}', recursive=True):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
                removed_count += 1
                print(f"  Removed: {item}")
            except Exception as e:
                print(f"  Could not remove {item}: {e}")

    if removed_count > 0:
        print(f"\n✓ Removed {removed_count} temporary files/directories")

    return removed_count

def main():
    """Run the complete cleanup process."""
    print("="*60)
    print("PATIENT ADVOCACY AGENT - PROJECT CLEANUP")
    print("="*60)
    print(f"Starting cleanup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Auto-confirm for script execution
    print("Starting automatic cleanup...")
    print()

    # Run cleanup steps
    create_directories()
    print()

    moved = move_files()
    print()

    removed = cleanup_temp_files()
    print()

    summary = create_project_summary()
    print()

    create_main_readme()
    print()

    # Print final summary
    print("="*60)
    print("CLEANUP COMPLETE")
    print("="*60)
    print(f"✓ Moved {moved} files to organized directories")
    print(f"✓ Removed {removed} temporary files")
    print(f"✓ Total files organized: {summary['total_files']}")
    print(f"✓ Project size: {summary['total_size_mb']} MB")
    print()
    print("Project structure has been organized!")
    print("See PROJECT_SUMMARY.json for details.")

if __name__ == "__main__":
    main()