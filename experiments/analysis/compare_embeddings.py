#!/usr/bin/env python3
"""
Compare clustering performance: Vanilla SigLIP vs Fine-tuned SigLIP

This script evaluates whether fine-tuning improved the embeddings for
medical skin condition clustering.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from collections import Counter
import json

import sys
sys.path.append('/home/sankar/patient_advocacy_agent/src')
from patient_advocacy_agent.embedder import SigLIPEmbedder


def generate_embeddings(embedder, metadata, image_dir, device, model_name="model"):
    """Generate embeddings for all images."""
    print(f"\nGenerating embeddings with {model_name}...")

    embeddings_list = []
    labels_list = []
    condition_names_list = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=model_name):
        image_path = image_dir / row['image_path']

        if not image_path.exists():
            continue

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = embedder.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)

            # Extract embedding
            with torch.no_grad():
                embedding = embedder.extract_image_features(pixel_values)
                embedding = embedding.squeeze(0).cpu().numpy()

            embeddings_list.append(embedding)
            labels_list.append(row['condition_label'])
            condition_names_list.append(row['condition'])

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)
    condition_names = np.array(condition_names_list)

    print(f"✓ Generated {len(embeddings)} embeddings (dim={embeddings.shape[1]})")

    return embeddings, labels, condition_names


def evaluate_clustering(embeddings, labels, condition_names, model_name="model"):
    """Evaluate clustering performance."""
    print(f"\n{'='*80}")
    print(f"CLUSTERING EVALUATION: {model_name}")
    print('='*80)

    n_clusters = len(np.unique(labels))

    # K-means clustering
    print(f"\nRunning K-means (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_pred = kmeans.fit_predict(embeddings)

    # Compute metrics
    silhouette = silhouette_score(embeddings, cluster_pred)
    ari = adjusted_rand_score(labels, cluster_pred)
    nmi = normalized_mutual_info_score(labels, cluster_pred)
    homogeneity = homogeneity_score(labels, cluster_pred)
    completeness = completeness_score(labels, cluster_pred)
    v_measure = v_measure_score(labels, cluster_pred)

    # Cluster purity
    cluster_purities = []
    perfect_clusters = 0
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_pred == cluster_id
        cluster_labels = labels[cluster_mask]

        if len(cluster_labels) == 0:
            continue

        most_common_label = Counter(cluster_labels).most_common(1)[0]
        purity = most_common_label[1] / len(cluster_labels)
        cluster_purities.append(purity)

        if purity == 1.0:
            perfect_clusters += 1

    avg_purity = np.mean(cluster_purities)

    print(f"\nClustering Quality Metrics:")
    print(f"  Silhouette Score:      {silhouette:.4f}")
    print(f"  Adjusted Rand Index:   {ari:.4f}")
    print(f"  Normalized Mutual Info: {nmi:.4f}")
    print(f"  Homogeneity:           {homogeneity:.4f}")
    print(f"  Completeness:          {completeness:.4f}")
    print(f"  V-Measure:             {v_measure:.4f}")
    print(f"  Average Cluster Purity: {avg_purity:.2%}")
    print(f"  Perfect Clusters:      {perfect_clusters}/{n_clusters}")

    # Per-condition analysis for top conditions
    print(f"\nTop 10 Conditions - Clustering Analysis:")
    condition_counts = Counter(condition_names)
    top_conditions = [cond for cond, count in condition_counts.most_common(10)]

    for condition in top_conditions:
        condition_mask = condition_names == condition
        condition_clusters = cluster_pred[condition_mask]

        unique_clusters = len(np.unique(condition_clusters))
        total_samples = len(condition_clusters)

        cluster_counts = Counter(condition_clusters)
        largest_cluster_size = cluster_counts.most_common(1)[0][1]
        clustering_ratio = largest_cluster_size / total_samples

        print(f"  {condition} ({total_samples} samples):")
        print(f"    Spread: {unique_clusters} clusters, {clustering_ratio:.1%} in largest")

    return {
        'silhouette': silhouette,
        'ari': ari,
        'nmi': nmi,
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure,
        'avg_purity': avg_purity,
        'perfect_clusters': perfect_clusters,
        'n_clusters': n_clusters
    }


def main():
    # Configuration
    METADATA_PATH = '/home/sankar/data/scin/real_labeled_metadata.csv'
    IMAGE_DIR = Path('/home/sankar/data/scin/images')
    FINETUNED_MODEL_PATH = '/home/sankar/models/embedder_real_labels/final/embedder_labeled_full.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*80)
    print("COMPARING VANILLA vs FINE-TUNED SIGLIP EMBEDDINGS")
    print("="*80)
    print(f"Device: {device}")
    print(f"Dataset: {METADATA_PATH}")

    # Load metadata
    print("\nLoading metadata...")
    metadata = pd.read_csv(METADATA_PATH)
    print(f"✓ Loaded {len(metadata)} images with {metadata['condition'].nunique()} conditions")

    # ========================================================================
    # 1. VANILLA SIGLIP (Original pre-trained model)
    # ========================================================================
    print("\n" + "="*80)
    print("1. VANILLA SIGLIP (Original Pre-trained)")
    print("="*80)

    vanilla_embedder = SigLIPEmbedder()
    vanilla_embedder.to(device)
    vanilla_embedder.eval()

    vanilla_embeddings, vanilla_labels, vanilla_condition_names = generate_embeddings(
        vanilla_embedder, metadata, IMAGE_DIR, device, "Vanilla SigLIP"
    )

    vanilla_results = evaluate_clustering(
        vanilla_embeddings, vanilla_labels, vanilla_condition_names, "VANILLA SIGLIP"
    )

    # ========================================================================
    # 2. FINE-TUNED SIGLIP (Trained on medical data)
    # ========================================================================
    print("\n" + "="*80)
    print("2. FINE-TUNED SIGLIP (Trained on Medical Skin Data)")
    print("="*80)

    finetuned_embedder = SigLIPEmbedder()
    checkpoint = torch.load(FINETUNED_MODEL_PATH, map_location=device)
    finetuned_embedder.load_state_dict(checkpoint['model_state_dict'])
    finetuned_embedder.to(device)
    finetuned_embedder.eval()

    finetuned_embeddings, finetuned_labels, finetuned_condition_names = generate_embeddings(
        finetuned_embedder, metadata, IMAGE_DIR, device, "Fine-tuned SigLIP"
    )

    finetuned_results = evaluate_clustering(
        finetuned_embeddings, finetuned_labels, finetuned_condition_names, "FINE-TUNED SIGLIP"
    )

    # ========================================================================
    # 3. COMPARISON SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    metrics = [
        ('Silhouette Score', 'silhouette'),
        ('Adjusted Rand Index', 'ari'),
        ('Normalized Mutual Info', 'nmi'),
        ('Homogeneity', 'homogeneity'),
        ('Completeness', 'completeness'),
        ('V-Measure', 'v_measure'),
        ('Average Cluster Purity', 'avg_purity')
    ]

    print(f"\n{'Metric':<30} {'Vanilla':<12} {'Fine-tuned':<12} {'Change':<12} {'Winner'}")
    print("-" * 80)

    for metric_name, metric_key in metrics:
        vanilla_val = vanilla_results[metric_key]
        finetuned_val = finetuned_results[metric_key]
        change = finetuned_val - vanilla_val
        pct_change = (change / vanilla_val * 100) if vanilla_val != 0 else 0

        if abs(change) < 0.001:
            winner = "Tie"
            symbol = "≈"
        elif change > 0:
            winner = "Fine-tuned ✓"
            symbol = "↑"
        else:
            winner = "Vanilla ✓"
            symbol = "↓"

        vanilla_str = f"{vanilla_val:.4f}"
        finetuned_str = f"{finetuned_val:.4f}"

        if metric_key == 'avg_purity':
            vanilla_str = f"{vanilla_val:.2%}"
            finetuned_str = f"{finetuned_val:.2%}"
            change_str = f"{symbol} {abs(pct_change):.1f}%"
        else:
            change_str = f"{symbol} {change:+.4f}"

        print(f"{metric_name:<30} {vanilla_str:<12} {finetuned_str:<12} {change_str:<12} {winner}")

    print("\n" + "-" * 80)
    print("Perfect Clusters:")
    print(f"  Vanilla:     {vanilla_results['perfect_clusters']}/{vanilla_results['n_clusters']}")
    print(f"  Fine-tuned:  {finetuned_results['perfect_clusters']}/{finetuned_results['n_clusters']}")

    # Overall verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    # Count wins
    vanilla_wins = 0
    finetuned_wins = 0
    ties = 0

    for _, metric_key in metrics:
        vanilla_val = vanilla_results[metric_key]
        finetuned_val = finetuned_results[metric_key]
        change = finetuned_val - vanilla_val

        if abs(change) < 0.001:
            ties += 1
        elif change > 0:
            finetuned_wins += 1
        else:
            vanilla_wins += 1

    print(f"\nScore: Vanilla {vanilla_wins} | Fine-tuned {finetuned_wins} | Ties {ties}")

    if finetuned_wins > vanilla_wins:
        print("\n✓ FINE-TUNING IMPROVED clustering performance")
        print("  The trained model creates better embeddings for medical diagnosis clustering.")
    elif vanilla_wins > finetuned_wins:
        print("\n✗ FINE-TUNING DEGRADED clustering performance")
        print("  The vanilla pre-trained model performs better for this task.")
        print("  Consider: more training data, different architecture, or hyperparameters.")
    else:
        print("\n~ FINE-TUNING HAD MINIMAL IMPACT")
        print("  Both models perform similarly for clustering medical diagnoses.")

    # Save results
    output_path = Path('/home/sankar/patient_advocacy_agent/embedding_comparison_results.json')
    results = {
        'vanilla': {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                   for k, v in vanilla_results.items()},
        'finetuned': {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                     for k, v in finetuned_results.items()},
        'summary': {
            'vanilla_wins': vanilla_wins,
            'finetuned_wins': finetuned_wins,
            'ties': ties
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
