#!/usr/bin/env python
"""
Build FAISS similarity index and RAG pipeline.

This script:
1. Loads the trained embedder
2. Extracts embeddings for all images
3. Builds FAISS similarity index
4. Creates RAG knowledge base
5. Saves everything for inference

Run with: uv run python build_index.py
"""

import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import json
import yaml

from patient_advocacy_agent import (
    SCINDataLoader,
    SigLIPEmbedder,
    SimilarityIndex,
    CaseRetriever,
    MedicalKnowledgeBase,
    RAGPipeline,
)

# Try to import langchain Document
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        class Document:
            def __init__(self, page_content: str, metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}

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


class IndexConfig:
    """Index building configuration."""

    def __init__(self, config_path: str = "config.yaml"):
        # Load config from YAML
        config = load_config(config_path)
        
        # Extract config sections
        data_config = config.get('data', {})
        model_config = config.get('models', {})
        embedder_config = model_config.get('embedder', {})
        embeddings_config = config.get('embeddings', {})
        
        # Paths - read from config.yaml with fallbacks
        self.data_dir = Path(data_config.get('scin_dir', './data/scin'))
        self.model_dir = Path(model_config.get('base_dir', './models'))
        self.embedder_path = Path(embedder_config.get('model_path', 
                                                       './models/embedder/final/embedder.pt'))
        self.index_dir = Path(model_config.get('similarity_index', {}).get('dir', 
                                                                            './models/similarity_index'))
        self.rag_dir = Path(model_config.get('rag_pipeline', {}).get('dir', 
                                                                      './models/rag_pipeline'))

        # Settings - read from config.yaml with fallbacks
        device_config = config.get('device', {})
        device_type = device_config.get('type', 'auto')
        
        if device_type == 'auto':
            self.device = self._get_device()
        else:
            self.device = device_type
            logger.info(f"Using device from config: {self.device}")
        
        self.batch_size = embeddings_config.get('batch_size', 32)
        self.num_workers = embeddings_config.get('num_workers', 0)  # Set to 0 for MPS
        self.use_gpu_index = torch.cuda.is_available() or torch.backends.mps.is_available()

    def _get_device(self) -> str:
        """Get best available device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"


def load_embedder(config: IndexConfig) -> tuple:
    """Load trained embedder."""
    print("\n" + "="*80)
    print("Loading Embedder")
    print("="*80)

    if not config.embedder_path.exists():
        print(f"✗ Embedder not found at {config.embedder_path}")
        print("Run: python train_embedder.py")
        return None, None

    try:
        embedder = SigLIPEmbedder.load(config.embedder_path)
        embedder = embedder.to(config.device)
        print(f"✓ Embedder loaded from {config.embedder_path}")

        # Load metadata
        metadata_path = config.data_dir / "metadata.csv"
        if metadata_path.exists():
            df = pd.read_csv(metadata_path)
            print(f"✓ Metadata loaded ({len(df)} images)")
            return embedder, df
        else:
            print(f"✗ Metadata not found at {metadata_path}")
            return None, None

    except Exception as e:
        print(f"✗ Failed to load embedder: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def extract_embeddings(
    config: IndexConfig,
    embedder,
    metadata_df: pd.DataFrame
) -> np.ndarray:
    """Extract embeddings for all images."""
    print("\n" + "="*80)
    print("Extracting Embeddings")
    print("="*80)

    data_loader = SCINDataLoader(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    try:
        dataloaders = data_loader.create_dataloaders()
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return None

    embeddings_list = []
    total_images = 0

    print(f"Extracting embeddings (batch_size={config.batch_size})...")

    try:
        # Use train + val + test data for index
        for split_name in ['train', 'val', 'test']:
            if split_name not in dataloaders:
                continue

            loader = dataloaders[split_name]
            split_embeddings = []

            for batch_idx, batch in enumerate(loader):
                try:
                    images = batch['image'].to(config.device)

                    with torch.no_grad():
                        embeddings = embedder.extract_image_features(images)

                    split_embeddings.append(embeddings.cpu().numpy())
                    total_images += len(images)

                    if (batch_idx + 1) % 5 == 0:
                        print(f"  {split_name}: {total_images} images processed")

                except Exception as e:
                    print(f"  Warning: Failed to process batch {batch_idx}: {e}")
                    continue

            embeddings_list.extend(split_embeddings)

        if not embeddings_list:
            print("✗ No embeddings extracted")
            return None

        embeddings = np.concatenate(embeddings_list).astype(np.float32)
        print(f"✓ Extracted embeddings for {len(embeddings)} images")
        print(f"  Shape: {embeddings.shape}")

        return embeddings

    except Exception as e:
        print(f"✗ Failed to extract embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_similarity_index(
    config: IndexConfig,
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame
) -> SimilarityIndex:
    """Build FAISS similarity index."""
    print("\n" + "="*80)
    print("Building Similarity Index")
    print("="*80)

    try:
        index = SimilarityIndex(
            embeddings=embeddings,
            metadata_df=metadata_df,
            use_gpu=config.use_gpu_index,
        )
        print(f"✓ Index created")

        # Save index
        config.index_dir.mkdir(parents=True, exist_ok=True)
        index.save(config.index_dir)
        print(f"✓ Index saved to {config.index_dir}")

        return index

    except Exception as e:
        print(f"✗ Failed to build index: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_rag_pipeline(
    config: IndexConfig,
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
) -> RAGPipeline:
    """Build RAG pipeline."""
    print("\n" + "="*80)
    print("Building RAG Pipeline")
    print("="*80)

    try:
        # Create case retriever
        print("Creating case retriever...")
        case_retriever = CaseRetriever(
            metadata_df=metadata_df,
            embeddings=embeddings,
        )
        print("✓ Case retriever created")

        # Create knowledge base with sample documents
        print("Creating knowledge base...")
        kb = MedicalKnowledgeBase()

        # Add sample medical documents
        sample_docs = [
            Document(
                page_content=(
                    "Eczema (atopic dermatitis) is a chronic inflammatory skin condition characterized by "
                    "itching, redness, and dryness. It commonly appears in children but can occur at any age. "
                    "Treatment focuses on skin barrier repair and managing inflammation using moisturizers and "
                    "topical corticosteroids. Triggers vary by individual and may include allergens, irritants, "
                    "stress, and environmental factors."
                ),
                metadata={'source': 'medical_db', 'condition': 'eczema', 'type': 'overview'}
            ),
            Document(
                page_content=(
                    "Psoriasis is a chronic autoimmune condition affecting skin characterized by thick, "
                    "scaly plaques often with clear boundaries. Common types include plaque psoriasis, "
                    "guttate psoriasis, and inverse psoriasis. Treatment options range from topical therapies "
                    "(corticosteroids, vitamin D analogues) to systemic treatments and biologics for moderate to "
                    "severe cases. Phototherapy may also be beneficial."
                ),
                metadata={'source': 'medical_db', 'condition': 'psoriasis', 'type': 'overview'}
            ),
            Document(
                page_content=(
                    "Contact dermatitis is an inflammatory skin reaction triggered by allergens or irritants. "
                    "It presents as localized rash, itching, and sometimes blistering. The condition can be "
                    "allergic (mediated by immune system) or irritant (chemical burn-like). Treatment includes "
                    "identifying and avoiding triggers, plus topical corticosteroids or antihistamines for symptom relief."
                ),
                metadata={'source': 'medical_db', 'condition': 'dermatitis', 'type': 'overview'}
            ),
            Document(
                page_content=(
                    "Acne is a common skin condition involving clogged pores, bacterial growth, and inflammation. "
                    "It affects sebaceous glands and typically appears on face, chest, and back. Treatment depends on "
                    "severity: mild cases respond to topical retinoids or benzoyl peroxide; moderate cases may need oral "
                    "antibiotics or hormonal therapy; severe cases may require isotretinoin. Proper skincare is essential."
                ),
                metadata={'source': 'medical_db', 'condition': 'acne', 'type': 'overview'}
            ),
            Document(
                page_content=(
                    "Fungal infections of the skin include conditions like tinea pedis (athlete's foot), tinea corporis "
                    "(ringworm), and candidiasis. They present with itching, redness, scaling, and sometimes discoloration. "
                    "Diagnosis often involves KOH preparation or culture. Treatment includes antifungal agents (topical for "
                    "localized infections, systemic for widespread disease) and hygiene measures to prevent recurrence."
                ),
                metadata={'source': 'medical_db', 'condition': 'fungal_infection', 'type': 'overview'}
            ),
        ]

        kb.add_documents(sample_docs)
        print(f"✓ Knowledge base created with {len(sample_docs)} documents")

        # Create RAG pipeline
        rag = RAGPipeline(case_retriever, kb)
        print("✓ RAG pipeline created")

        # Save RAG
        config.rag_dir.mkdir(parents=True, exist_ok=True)
        rag.save(config.rag_dir)
        print(f"✓ RAG pipeline saved to {config.rag_dir}")

        return rag

    except Exception as e:
        print(f"✗ Failed to build RAG pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_summary(config: IndexConfig, embeddings: np.ndarray, metadata_df: pd.DataFrame):
    """Save summary of index building."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'embedder_path': str(config.embedder_path),
        'index_dir': str(config.index_dir),
        'rag_dir': str(config.rag_dir),
        'embeddings': {
            'shape': embeddings.shape,
            'dtype': str(embeddings.dtype),
            'size_mb': embeddings.nbytes / (1024 ** 2),
        },
        'dataset': {
            'total_images': len(metadata_df),
            'conditions': len(metadata_df['condition'].unique()),
            'condition_distribution': metadata_df['condition'].value_counts().to_dict(),
        },
        'device': config.device,
    }

    summary_path = config.model_dir / "index_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Index summary saved to {summary_path}")
    print(f"\nIndex Statistics:")
    print(f"  - Total images: {summary['dataset']['total_images']}")
    print(f"  - Conditions: {summary['dataset']['conditions']}")
    print(f"  - Embeddings size: {summary['embeddings']['size_mb']:.1f} MB")


def main():
    """Main function."""
    print("\n" + "="*80)
    print("Build FAISS Index and RAG Pipeline")
    print("="*80)

    config = IndexConfig()

    print(f"\nConfiguration (from config.yaml):")
    print(f"  Data dir: {config.data_dir}")
    print(f"  Embedder: {config.embedder_path}")
    print(f"  Index dir: {config.index_dir}")
    print(f"  RAG dir: {config.rag_dir}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Device: {config.device}")

    # Load embedder
    embedder, metadata_df = load_embedder(config)
    if embedder is None or metadata_df is None:
        return 1

    # Extract embeddings
    embeddings = extract_embeddings(config, embedder, metadata_df)
    if embeddings is None:
        return 1

    # Build similarity index
    index = build_similarity_index(config, embeddings, metadata_df)
    if index is None:
        return 1

    # Build RAG pipeline
    rag = build_rag_pipeline(config, embeddings, metadata_df)
    if rag is None:
        return 1

    # Save summary
    save_summary(config, embeddings, metadata_df)

    print("\n" + "="*80)
    print("Index Building Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Create assessment API:")
    print("   uv run python create_api.py")
    print("\n2. Run assessments:")
    print("   uv run python example_usage.py")
    print("\n3. Deploy:")
    print("   uv run python -m uvicorn api:app")

    return 0


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
