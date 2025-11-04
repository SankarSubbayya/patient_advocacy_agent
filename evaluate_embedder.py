#!/usr/bin/env python
"""
Evaluate and compare vanilla SigLIP vs fine-tuned embedder performance.

This script:
1. Loads both vanilla SigLIP and fine-tuned models
2. Extracts embeddings for test set
3. Computes retrieval metrics (Recall@K, MRR, Precision@K, etc.)
4. Generates comparison report with visualizations

Run with: uv run python evaluate_embedder.py
"""

import torch
import numpy as np
import faiss
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

from patient_advocacy_agent import (
    SCINDataLoader,
    SigLIPEmbedder,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Evaluates retrieval performance of embedders."""

    def __init__(self, embedder, device: str = 'cuda'):
        """Initialize evaluator with an embedder model."""
        self.embedder = embedder
        self.device = device
        self.embedder.to(device)
        self.embedder.eval()

    @torch.no_grad()
    def extract_embeddings(self, images: torch.Tensor) -> np.ndarray:
        """Extract embeddings from images."""
        images = images.to(self.device)
        embeddings = self.embedder.extract_image_features(images)
        return embeddings.cpu().numpy()

    def build_retrieval_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Build FAISS index for fast retrieval."""
        # Convert to float32 for FAISS
        embeddings = embeddings.astype(np.float32)

        # Create L2 index (we'll convert distances to similarity later)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        return index, embeddings

    def compute_retrieval_metrics(
        self,
        index: faiss.IndexFlatL2,
        query_embeddings: np.ndarray,
        query_labels: np.ndarray,
        corpus_labels: np.ndarray,
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics.

        Returns:
            Dictionary with metrics like Recall@K, Precision@K, MRR, etc.
        """
        query_embeddings = query_embeddings.astype(np.float32)

        # Search for nearest neighbors
        distances, indices = index.search(query_embeddings, max(k_values))

        # Convert L2 distances to similarity (1 / (1 + distance))
        similarities = 1.0 / (1.0 + distances)

        metrics = {}

        # Compute Recall@K and Precision@K
        for k in k_values:
            top_k_indices = indices[:, :k]
            top_k_labels = corpus_labels[top_k_indices]

            # Recall@K: % of correct items in top-K
            recall_k = []
            precision_k = []

            for i in range(len(query_labels)):
                correct = np.sum(top_k_labels[i] == query_labels[i])
                recall_k.append(correct / k)
                precision_k.append(correct / k)  # Since each query should have at least 1 occurrence

            metrics[f'Recall@{k}'] = np.mean(recall_k)
            metrics[f'Precision@{k}'] = np.mean(precision_k)

        # Compute Mean Reciprocal Rank (MRR)
        mrr_scores = []
        for i in range(len(query_labels)):
            # Find first occurrence of correct label in top-K results
            correct_positions = np.where(top_k_labels[i] == query_labels[i])[0]
            if len(correct_positions) > 0:
                mrr_scores.append(1.0 / (correct_positions[0] + 1))
            else:
                mrr_scores.append(0.0)

        metrics['MRR'] = np.mean(mrr_scores)

        # Compute Retrieval Accuracy (is first result correct?)
        top_1_labels = corpus_labels[indices[:, 0]]
        metrics['Accuracy@1'] = np.mean(top_1_labels == query_labels)

        # Compute Hit Rate (is correct label in top-K?)
        for k in k_values:
            hit_rate = []
            for i in range(len(query_labels)):
                if query_labels[i] in top_k_labels[i]:
                    hit_rate.append(1.0)
                else:
                    hit_rate.append(0.0)
            metrics[f'HitRate@{k}'] = np.mean(hit_rate)

        return metrics


def load_vanilla_siglip(device: str = 'cuda') -> SigLIPEmbedder:
    """Load vanilla SigLIP model (no fine-tuning)."""
    logger.info("Loading vanilla SigLIP model...")
    embedder = SigLIPEmbedder(
        model_name='google/siglip-base-patch16-224',
        freeze_backbone=True  # Don't train, just use as-is
    )
    embedder.to(device)
    embedder.eval()
    return embedder


def load_finetuned_model(model_path: Path, device: str = 'cuda') -> SigLIPEmbedder:
    """Load fine-tuned model from checkpoint."""
    logger.info(f"Loading fine-tuned model from {model_path}...")
    embedder = SigLIPEmbedder.load(model_path)
    embedder.to(device)
    embedder.eval()
    return embedder


def evaluate(
    vanilla_embedder: SigLIPEmbedder,
    finetuned_embedder: SigLIPEmbedder,
    test_loader,
    device: str = 'cuda'
) -> Tuple[Dict, Dict]:
    """Evaluate both embedders on test set."""

    logger.info("\nExtracting test set embeddings...")

    # Collect all test data
    test_images = []
    test_labels = []
    test_conditions = []

    for batch in test_loader:
        test_images.append(batch['image'])
        test_labels.extend(batch['label'])
        test_conditions.extend(batch['condition'])

    test_images = torch.cat(test_images)
    test_labels = np.array(test_labels)

    logger.info(f"Test set size: {len(test_labels)} images, {len(np.unique(test_labels))} conditions")

    # Extract embeddings using vanilla model
    logger.info("Extracting vanilla SigLIP embeddings...")
    vanilla_evaluator = RetrievalEvaluator(vanilla_embedder, device)
    vanilla_embeddings = vanilla_evaluator.extract_embeddings(test_images)

    # Extract embeddings using fine-tuned model
    logger.info("Extracting fine-tuned embeddings...")
    finetuned_evaluator = RetrievalEvaluator(finetuned_embedder, device)
    finetuned_embeddings = finetuned_evaluator.extract_embeddings(test_images)

    # Build indices (use test set as corpus for retrieval)
    logger.info("Building retrieval indices...")
    vanilla_index, vanilla_corpus = vanilla_evaluator.build_retrieval_index(vanilla_embeddings)
    finetuned_index, finetuned_corpus = finetuned_evaluator.build_retrieval_index(finetuned_embeddings)

    # Compute metrics
    logger.info("Computing retrieval metrics...")
    k_values = [1, 5, 10, 20]

    vanilla_metrics = vanilla_evaluator.compute_retrieval_metrics(
        vanilla_index,
        vanilla_embeddings,
        test_labels,
        test_labels,
        k_values=k_values
    )

    finetuned_metrics = finetuned_evaluator.compute_retrieval_metrics(
        finetuned_index,
        finetuned_embeddings,
        test_labels,
        test_labels,
        k_values=k_values
    )

    return vanilla_metrics, finetuned_metrics


def create_comparison_report(
    vanilla_metrics: Dict,
    finetuned_metrics: Dict,
    output_dir: Path = Path('./models/embedder/final')
) -> None:
    """Create and save comparison report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / 'EVALUATION_REPORT.txt'

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EMBEDDING MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        f.write("METRICS COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<25} {'Vanilla SigLIP':<20} {'Fine-tuned':<20} {'Improvement':<15}\n")
        f.write("-"*80 + "\n")

        improvements = {}

        for metric_name in sorted(vanilla_metrics.keys()):
            vanilla_val = vanilla_metrics[metric_name]
            finetuned_val = finetuned_metrics[metric_name]

            # Calculate improvement (percentage points for 0-1 metrics)
            improvement = (finetuned_val - vanilla_val) * 100
            improvements[metric_name] = improvement

            improvement_str = f"{improvement:+.2f}pp" if abs(improvement) < 10 else f"{improvement:+.1f}pp"

            f.write(f"{metric_name:<25} {vanilla_val:<20.4f} {finetuned_val:<20.4f} {improvement_str:<15}\n")

        # Summary
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")

        # Average improvement
        avg_improvement = np.mean(list(improvements.values()))
        f.write(f"\nAverage improvement: {avg_improvement:+.2f} percentage points\n")

        # Best and worst metrics
        best_metric = max(improvements.items(), key=lambda x: x[1])
        worst_metric = min(improvements.items(), key=lambda x: x[1])

        f.write(f"\nBest improvement: {best_metric[0]} (+{best_metric[1]:.2f}pp)\n")
        f.write(f"Worst performance: {worst_metric[0]} ({worst_metric[1]:+.2f}pp)\n")

        # Interpretation
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n")

        if avg_improvement > 2:
            f.write("\n✓ POSITIVE: Fine-tuned model shows SIGNIFICANT improvement over vanilla SigLIP.\n")
            f.write("  The model has learned skin condition-specific features from the SCIN dataset.\n")
        elif avg_improvement > 0:
            f.write("\n✓ SLIGHT IMPROVEMENT: Fine-tuned model slightly outperforms vanilla SigLIP.\n")
            f.write("  Improvement is modest but consistent across metrics.\n")
        elif avg_improvement > -2:
            f.write("\n△ MARGINAL: Fine-tuned model performs similarly to vanilla SigLIP.\n")
            f.write("  Check if more training data or epochs could help.\n")
        else:
            f.write("\n✗ REGRESSION: Fine-tuned model underperforms vanilla SigLIP.\n")
            f.write("  Consider: more training data, different learning rate, or more epochs.\n")

        f.write("\nKey Metrics Explanation:\n")
        f.write("  - Recall@K: % of correct items found in top-K results\n")
        f.write("  - Precision@K: % of top-K results that are actually correct\n")
        f.write("  - MRR: Average position of first correct result (higher is better)\n")
        f.write("  - Accuracy@1: Is the top result correct?\n")
        f.write("  - HitRate@K: Did we find the correct condition in top-K?\n")

    logger.info(f"Report saved to {report_path}")
    return report_path


def create_visualization(
    vanilla_metrics: Dict,
    finetuned_metrics: Dict,
    output_dir: Path = Path('./models/embedder/final')
) -> None:
    """Create visualization comparing metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter metrics that are comparable
    comparable_metrics = [m for m in vanilla_metrics.keys()
                         if m in finetuned_metrics and 'Recall' in m or 'Precision' in m or 'Accuracy' in m or 'MRR' in m or 'HitRate' in m]

    vanilla_values = [vanilla_metrics[m] for m in comparable_metrics]
    finetuned_values = [finetuned_metrics[m] for m in comparable_metrics]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Comparison plot
    x = np.arange(len(comparable_metrics))
    width = 0.35

    ax1.bar(x - width/2, vanilla_values, width, label='Vanilla SigLIP', alpha=0.8)
    ax1.bar(x + width/2, finetuned_values, width, label='Fine-tuned', alpha=0.8)

    ax1.set_ylabel('Score')
    ax1.set_title('Metric Comparison: Vanilla vs Fine-tuned')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparable_metrics, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.1])

    # Improvement plot
    improvements = [(finetuned_values[i] - vanilla_values[i]) * 100
                    for i in range(len(comparable_metrics))]
    colors = ['green' if x > 0 else 'red' for x in improvements]

    ax2.bar(range(len(comparable_metrics)), improvements, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_ylabel('Improvement (percentage points)')
    ax2.set_title('Performance Improvement')
    ax2.set_xticks(range(len(comparable_metrics)))
    ax2.set_xticklabels(comparable_metrics, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    viz_path = output_dir / 'evaluation_comparison.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to {viz_path}")
    plt.close()


def main():
    """Main evaluation function."""
    print("\n" + "="*80)
    print("EMBEDDING MODEL EVALUATION")
    print("="*80)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Setup config - use absolute paths
    data_dir = Path('/home/sankar/data/scin')
    model_dir = Path('/home/sankar/models/embedder')
    finetuned_path = model_dir / 'final' / 'embedder.pt'

    # Check paths
    if not data_dir.exists():
        print(f"\n✗ Data directory not found: {data_dir}")
        print("Run: python download_scin_dataset.py")
        return 1

    if not finetuned_path.exists():
        print(f"\n✗ Fine-tuned model not found: {finetuned_path}")
        print("Run: python train_embedder.py")
        return 1

    print(f"✓ Data dir: {data_dir}")
    print(f"✓ Fine-tuned model: {finetuned_path}")

    # Load dataset
    print("\n" + "="*80)
    print("Loading Dataset")
    print("="*80)

    try:
        data_loader = SCINDataLoader(
            data_dir=data_dir,
            batch_size=32,
            num_workers=0,
            test_split=0.2,
            val_split=0.1,
        )
        dataloaders = data_loader.create_dataloaders()

        print(f"✓ Dataset loaded")
        print(f"  - Test: {len(data_loader.test_dataset)} images")
        print(f"  - Conditions: {data_loader.get_num_classes()}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return 1

    # Load models
    print("\n" + "="*80)
    print("Loading Models")
    print("="*80)

    try:
        vanilla_embedder = load_vanilla_siglip(device)
        finetuned_embedder = load_finetuned_model(finetuned_path, device)
        print("✓ Both models loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Evaluate
    print("\n" + "="*80)
    print("Evaluating Models")
    print("="*80)

    try:
        vanilla_metrics, finetuned_metrics = evaluate(
            vanilla_embedder,
            finetuned_embedder,
            dataloaders['test'],
            device=device
        )

        print("\n✓ Evaluation complete")

        # Display results
        print("\n" + "-"*80)
        print("QUICK RESULTS")
        print("-"*80)
        print(f"{'Metric':<20} {'Vanilla':<15} {'Fine-tuned':<15} {'Improvement':<15}")
        print("-"*80)

        for metric in sorted(vanilla_metrics.keys()):
            vanilla_val = vanilla_metrics[metric]
            finetuned_val = finetuned_metrics[metric]
            improvement = (finetuned_val - vanilla_val) * 100

            improvement_str = f"{improvement:+.2f}pp"
            print(f"{metric:<20} {vanilla_val:<15.4f} {finetuned_val:<15.4f} {improvement_str:<15}")

    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Generate report
    print("\n" + "="*80)
    print("Generating Report")
    print("="*80)

    try:
        report_path = create_comparison_report(vanilla_metrics, finetuned_metrics)
        create_visualization(vanilla_metrics, finetuned_metrics)

        print(f"\n✓ Report saved")
        print(f"  - {report_path}")
        print(f"  - {Path(report_path).parent / 'evaluation_comparison.png'}")
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print("\nCheck the evaluation report for detailed analysis.")

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
