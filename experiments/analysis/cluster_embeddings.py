#!/usr/bin/env python3
"""
Clustering Analysis for Embedding Validation

Test the quality of trained embeddings by clustering and measuring how well
images of the same diagnosis cluster together.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from collections import Counter, defaultdict
import json

import sys
sys.path.append('/home/sankar/patient_advocacy_agent/src')
from patient_advocacy_agent.embedder import SigLIPEmbedder

class EmbeddingClusterAnalyzer:
    def __init__(
        self,
        model_path: str,
        metadata_path: str,
        image_dir: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.image_dir = Path(image_dir)

        # Load metadata
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)
        self.metadata = pd.read_csv(metadata_path)
        print(f"Loaded {len(self.metadata)} images with {self.metadata['condition'].nunique()} conditions")

        # Load model
        print("\nLoading embedding model...")
        self.embedder = SigLIPEmbedder()
        checkpoint = torch.load(model_path, map_location=device)
        self.embedder.load_state_dict(checkpoint['model_state_dict'])
        self.embedder.to(device)
        self.embedder.eval()
        print(f"✓ Model loaded from {model_path}")

        self.embeddings = None
        self.labels = None
        self.condition_names = None

    def generate_embeddings(self, max_samples: int = None):
        """Generate embeddings for all images."""
        print("\n" + "="*80)
        print("GENERATING EMBEDDINGS")
        print("="*80)

        if max_samples:
            metadata_subset = self.metadata.sample(n=min(max_samples, len(self.metadata)), random_state=42)
        else:
            metadata_subset = self.metadata

        embeddings_list = []
        labels_list = []
        condition_names_list = []

        print(f"Processing {len(metadata_subset)} images...")

        for _, row in tqdm(metadata_subset.iterrows(), total=len(metadata_subset)):
            image_path = self.image_dir / row['image_path']

            if not image_path.exists():
                continue

            try:
                # Load and preprocess image
                image = Image.open(image_path).convert('RGB')
                inputs = self.embedder.processor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)

                # Extract embedding
                with torch.no_grad():
                    embedding = self.embedder.extract_image_features(pixel_values)
                    embedding = embedding.squeeze(0).cpu().numpy()

                embeddings_list.append(embedding)
                labels_list.append(row['condition_label'])
                condition_names_list.append(row['condition'])

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        self.embeddings = np.array(embeddings_list)
        self.labels = np.array(labels_list)
        self.condition_names = np.array(condition_names_list)

        print(f"\n✓ Generated {len(self.embeddings)} embeddings")
        print(f"  Embedding dimension: {self.embeddings.shape[1]}")
        print(f"  Unique conditions: {len(np.unique(self.labels))}")

        return self.embeddings, self.labels, self.condition_names

    def kmeans_clustering(self, n_clusters: int = None):
        """Perform K-means clustering."""
        if n_clusters is None:
            n_clusters = len(np.unique(self.labels))

        print("\n" + "="*80)
        print(f"K-MEANS CLUSTERING (k={n_clusters})")
        print("="*80)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_pred = kmeans.fit_predict(self.embeddings)

        # Compute metrics
        silhouette = silhouette_score(self.embeddings, cluster_pred)
        ari = adjusted_rand_score(self.labels, cluster_pred)
        nmi = normalized_mutual_info_score(self.labels, cluster_pred)
        homogeneity = homogeneity_score(self.labels, cluster_pred)
        completeness = completeness_score(self.labels, cluster_pred)
        v_measure = v_measure_score(self.labels, cluster_pred)

        print(f"\nClustering Quality Metrics:")
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
        print(f"  Adjusted Rand Index: {ari:.4f} (higher is better, range: -1 to 1)")
        print(f"  Normalized Mutual Info: {nmi:.4f} (higher is better, range: 0 to 1)")
        print(f"  Homogeneity: {homogeneity:.4f} (higher is better, range: 0 to 1)")
        print(f"  Completeness: {completeness:.4f} (higher is better, range: 0 to 1)")
        print(f"  V-Measure: {v_measure:.4f} (higher is better, range: 0 to 1)")

        # Analyze cluster purity
        print("\n" + "-"*80)
        print("CLUSTER PURITY ANALYSIS")
        print("-"*80)

        cluster_purity = self.compute_cluster_purity(cluster_pred)
        print(f"\nAverage cluster purity: {cluster_purity['average_purity']:.2%}")
        print(f"Perfect clusters (100% purity): {cluster_purity['perfect_clusters']}/{n_clusters}")

        return {
            'predictions': cluster_pred,
            'silhouette': silhouette,
            'ari': ari,
            'nmi': nmi,
            'homogeneity': homogeneity,
            'completeness': completeness,
            'v_measure': v_measure,
            'cluster_purity': cluster_purity
        }

    def compute_cluster_purity(self, cluster_pred):
        """Compute purity of each cluster."""
        n_clusters = len(np.unique(cluster_pred))
        cluster_purities = []
        perfect_clusters = 0

        for cluster_id in range(n_clusters):
            # Get all true labels in this cluster
            cluster_mask = cluster_pred == cluster_id
            cluster_labels = self.labels[cluster_mask]

            if len(cluster_labels) == 0:
                continue

            # Most common label in this cluster
            most_common_label = Counter(cluster_labels).most_common(1)[0]
            purity = most_common_label[1] / len(cluster_labels)
            cluster_purities.append(purity)

            if purity == 1.0:
                perfect_clusters += 1

        return {
            'average_purity': np.mean(cluster_purities),
            'purities': cluster_purities,
            'perfect_clusters': perfect_clusters
        }

    def analyze_condition_clustering(self, cluster_pred, top_n: int = 10):
        """Analyze how well each condition clusters together."""
        print("\n" + "="*80)
        print(f"PER-CONDITION CLUSTERING ANALYSIS (Top {top_n})")
        print("="*80)

        # Count samples per condition
        condition_counts = Counter(self.condition_names)
        top_conditions = [cond for cond, count in condition_counts.most_common(top_n)]

        condition_metrics = []

        for condition in top_conditions:
            # Get all samples of this condition
            condition_mask = self.condition_names == condition
            condition_clusters = cluster_pred[condition_mask]

            # How many clusters does this condition span?
            unique_clusters = len(np.unique(condition_clusters))
            total_samples = len(condition_clusters)

            # What's the largest cluster for this condition?
            cluster_counts = Counter(condition_clusters)
            largest_cluster_size = cluster_counts.most_common(1)[0][1]
            clustering_ratio = largest_cluster_size / total_samples

            condition_metrics.append({
                'condition': condition,
                'total_samples': total_samples,
                'num_clusters': unique_clusters,
                'largest_cluster_pct': clustering_ratio
            })

            print(f"\n{condition} ({total_samples} samples):")
            print(f"  Spread across {unique_clusters} clusters")
            print(f"  {largest_cluster_size}/{total_samples} ({clustering_ratio:.1%}) in largest cluster")

        return condition_metrics

    def visualize_tsne(self, cluster_pred, output_path: str, top_n_conditions: int = 10):
        """Create t-SNE visualization."""
        print("\n" + "="*80)
        print("CREATING t-SNE VISUALIZATION")
        print("="*80)

        print("Computing t-SNE (this may take a few minutes)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(self.embeddings)

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 10))

        # 1. Color by true condition (top N conditions only)
        ax1 = plt.subplot(1, 2, 1)
        condition_counts = Counter(self.condition_names)
        top_conditions = [cond for cond, count in condition_counts.most_common(top_n_conditions)]

        # Plot background (other conditions) in gray
        other_mask = ~np.isin(self.condition_names, top_conditions)
        ax1.scatter(embeddings_2d[other_mask, 0], embeddings_2d[other_mask, 1],
                   c='lightgray', alpha=0.3, s=20, label='Other conditions')

        # Plot top conditions with distinct colors
        colors = plt.cm.tab20(np.linspace(0, 1, top_n_conditions))
        for i, condition in enumerate(top_conditions):
            mask = self.condition_names == condition
            count = np.sum(mask)
            ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[colors[i]], alpha=0.7, s=30, label=f'{condition} (n={count})')

        ax1.set_title(f't-SNE Visualization (Top {top_n_conditions} Conditions by Frequency)', fontsize=12, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')

        # 2. Color by predicted cluster
        ax2 = plt.subplot(1, 2, 2)
        n_clusters = len(np.unique(cluster_pred))
        cluster_colors = plt.cm.tab20(np.linspace(0, 1, min(n_clusters, 20)))

        for cluster_id in range(min(n_clusters, 20)):  # Limit to 20 for visualization
            mask = cluster_pred == cluster_id
            count = np.sum(mask)
            ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[cluster_colors[cluster_id]], alpha=0.7, s=30, label=f'Cluster {cluster_id} (n={count})')

        ax2.set_title(f't-SNE Visualization (K-means Clusters, k={n_clusters})', fontsize=12, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {output_path}")

        return embeddings_2d

    def create_confusion_heatmap(self, cluster_pred, output_path: str, top_n: int = 20):
        """Create heatmap showing cluster-condition relationships."""
        print("\n" + "="*80)
        print("CREATING CLUSTER-CONDITION HEATMAP")
        print("="*80)

        # Get top N conditions by frequency
        condition_counts = Counter(self.condition_names)
        top_conditions = [cond for cond, count in condition_counts.most_common(top_n)]

        # Create matrix: rows = conditions, cols = clusters
        n_clusters = len(np.unique(cluster_pred))
        matrix = np.zeros((len(top_conditions), n_clusters))

        for i, condition in enumerate(top_conditions):
            condition_mask = self.condition_names == condition
            condition_clusters = cluster_pred[condition_mask]

            for cluster_id in range(n_clusters):
                count = np.sum(condition_clusters == cluster_id)
                matrix[i, cluster_id] = count

        # Normalize by row (show percentage of each condition in each cluster)
        matrix_pct = matrix / matrix.sum(axis=1, keepdims=True) * 100

        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(12, n_clusters * 0.5), max(8, len(top_conditions) * 0.4)))
        sns.heatmap(matrix_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                   xticklabels=[f'C{i}' for i in range(n_clusters)],
                   yticklabels=top_conditions,
                   cbar_kws={'label': '% of Condition in Cluster'},
                   ax=ax)

        ax.set_title(f'Cluster Distribution for Top {top_n} Conditions\n(% of each condition assigned to each cluster)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Cluster ID', fontsize=11)
        ax.set_ylabel('Medical Condition', fontsize=11)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Heatmap saved to {output_path}")


def main():
    # Configuration
    MODEL_PATH = '/home/sankar/models/embedder_real_labels/final/embedder_labeled_full.pt'
    METADATA_PATH = '/home/sankar/data/scin/real_labeled_metadata.csv'
    IMAGE_DIR = '/home/sankar/data/scin/images'
    OUTPUT_DIR = Path('/home/sankar/patient_advocacy_agent')

    # Initialize analyzer
    analyzer = EmbeddingClusterAnalyzer(
        model_path=MODEL_PATH,
        metadata_path=METADATA_PATH,
        image_dir=IMAGE_DIR
    )

    # Generate embeddings (use subset for speed)
    # Use all data for comprehensive analysis, or set max_samples=1000 for quick test
    embeddings, labels, condition_names = analyzer.generate_embeddings(max_samples=None)

    # Perform K-means clustering
    n_conditions = len(np.unique(labels))
    kmeans_results = analyzer.kmeans_clustering(n_clusters=n_conditions)

    # Analyze per-condition clustering
    condition_metrics = analyzer.analyze_condition_clustering(
        kmeans_results['predictions'],
        top_n=20
    )

    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # t-SNE visualization
    tsne_output = OUTPUT_DIR / 'embedding_tsne_visualization.png'
    analyzer.visualize_tsne(
        kmeans_results['predictions'],
        output_path=str(tsne_output),
        top_n_conditions=15
    )

    # Heatmap
    heatmap_output = OUTPUT_DIR / 'embedding_cluster_heatmap.png'
    analyzer.create_confusion_heatmap(
        kmeans_results['predictions'],
        output_path=str(heatmap_output),
        top_n=20
    )

    # Save summary report
    print("\n" + "="*80)
    print("SAVING SUMMARY REPORT")
    print("="*80)

    report = {
        'model_path': MODEL_PATH,
        'num_samples': len(embeddings),
        'num_conditions': n_conditions,
        'embedding_dimension': embeddings.shape[1],
        'clustering_metrics': {
            'silhouette_score': float(kmeans_results['silhouette']),
            'adjusted_rand_index': float(kmeans_results['ari']),
            'normalized_mutual_info': float(kmeans_results['nmi']),
            'homogeneity': float(kmeans_results['homogeneity']),
            'completeness': float(kmeans_results['completeness']),
            'v_measure': float(kmeans_results['v_measure']),
            'average_cluster_purity': float(kmeans_results['cluster_purity']['average_purity']),
            'perfect_clusters': int(kmeans_results['cluster_purity']['perfect_clusters'])
        },
        'top_conditions': condition_metrics
    }

    report_path = OUTPUT_DIR / 'embedding_clustering_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Report saved to {report_path}")

    # Print final summary
    print("\n" + "="*80)
    print("CLUSTERING ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nKey Findings:")
    print(f"  • Analyzed {len(embeddings)} images across {n_conditions} conditions")
    print(f"  • Silhouette Score: {kmeans_results['silhouette']:.4f}")
    print(f"  • NMI (label agreement): {kmeans_results['nmi']:.4f}")
    print(f"  • Average Cluster Purity: {kmeans_results['cluster_purity']['average_purity']:.2%}")
    print(f"\nInterpretation:")
    if kmeans_results['silhouette'] > 0.3:
        print("  ✓ GOOD: Embeddings show clear cluster structure")
    elif kmeans_results['silhouette'] > 0.1:
        print("  ~ MODERATE: Some cluster structure exists")
    else:
        print("  ✗ WEAK: Limited cluster separation")

    if kmeans_results['nmi'] > 0.5:
        print("  ✓ GOOD: Clusters align well with medical diagnoses")
    elif kmeans_results['nmi'] > 0.3:
        print("  ~ MODERATE: Some alignment with diagnoses")
    else:
        print("  ✗ WEAK: Poor alignment with diagnoses")

    print(f"\nOutputs:")
    print(f"  • t-SNE plot: {tsne_output}")
    print(f"  • Heatmap: {heatmap_output}")
    print(f"  • Report: {report_path}")
    print()


if __name__ == '__main__':
    main()
