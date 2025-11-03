"""Clustering system for finding similar skin condition cases."""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import faiss
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ClusterResult(BaseModel):
    """Result of clustering analysis."""
    case_id: str
    condition: str
    similarity_score: float
    cluster_id: int


class SimilarityIndex:
    """Index for fast similarity search of skin condition images."""

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata_df: pd.DataFrame,
        use_gpu: bool = False
    ):
        """
        Initialize similarity index.

        Args:
            embeddings: Image embeddings array (N, D)
            metadata_df: DataFrame with image metadata
            use_gpu: Whether to use GPU for indexing
        """
        self.embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0

        # Build FAISS index
        self.index = self._build_index()

    def _build_index(self) -> faiss.Index:
        """Build FAISS index for fast similarity search."""
        d = self.embeddings.shape[1]

        # Use IVF (Inverted File) index for faster search on large datasets
        nlist = min(100, max(1, self.embeddings.shape[0] // 10))
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        # Train the index
        index.train(self.embeddings)

        # Add vectors to the index
        index.add(self.embeddings)

        if self.use_gpu:
            index = faiss.index_cpu_to_all_gpus(index)

        logger.info(f"Built FAISS index with {len(self.embeddings)} embeddings")
        return index

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[ClusterResult]:
        """
        Search for top-k similar images.

        Args:
            query_embedding: Query embedding (D,)
            k: Number of results to return
            threshold: Optional similarity threshold (L2 distance)

        Returns:
            List of top-k similar cases
        """
        query_embedding = np.ascontiguousarray(
            query_embedding.reshape(1, -1).astype(np.float32)
        )

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid result
                continue

            if threshold is not None and dist > threshold:
                continue

            # Convert L2 distance to similarity score (higher is better)
            similarity_score = 1.0 / (1.0 + float(dist))

            metadata = self.metadata_df.iloc[idx]
            result = ClusterResult(
                case_id=metadata.get('image_id', f'case_{idx}'),
                condition=metadata.get('condition', 'unknown'),
                similarity_score=similarity_score,
                cluster_id=int(idx)
            )
            results.append(result)

        return results

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5
    ) -> List[List[ClusterResult]]:
        """
        Search for top-k similar images for multiple queries.

        Args:
            query_embeddings: Query embeddings (B, D)
            k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        query_embeddings = np.ascontiguousarray(query_embeddings.astype(np.float32))
        distances, indices = self.index.search(query_embeddings, k)

        all_results = []
        for query_idx, (dists, inds) in enumerate(zip(distances, indices)):
            results = []
            for dist, idx in zip(dists, inds):
                if idx == -1:
                    continue

                similarity_score = 1.0 / (1.0 + float(dist))
                metadata = self.metadata_df.iloc[idx]
                result = ClusterResult(
                    case_id=metadata.get('image_id', f'case_{idx}'),
                    condition=metadata.get('condition', 'unknown'),
                    similarity_score=similarity_score,
                    cluster_id=int(idx)
                )
                results.append(result)
            all_results.append(results)

        return all_results

    def save(self, path: Path) -> None:
        """Save the index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss_index.bin"))

        # Save metadata
        self.metadata_df.to_csv(path / "metadata.csv", index=False)

        logger.info(f"Index saved to {path}")

    @classmethod
    def load(cls, path: Path, use_gpu: bool = False) -> "SimilarityIndex":
        """Load index from disk."""
        path = Path(path)

        # Load FAISS index
        index = faiss.read_index(str(path / "faiss_index.bin"))

        # Load metadata
        metadata_df = pd.read_csv(path / "metadata.csv")

        # Reconstruct embeddings from index
        embeddings = index.reconstruct_n(0, index.ntotal)

        obj = cls.__new__(cls)
        obj.embeddings = embeddings
        obj.metadata_df = metadata_df
        obj.use_gpu = use_gpu
        obj.index = index

        if use_gpu and faiss.get_num_gpus() > 0:
            obj.index = faiss.index_cpu_to_all_gpus(index)

        logger.info(f"Index loaded from {path}")
        return obj


class ImageClusterer:
    """Cluster images based on embeddings."""

    def __init__(self, n_clusters: int = 20):
        """
        Initialize clusterer.

        Args:
            n_clusters: Number of clusters
        """
        self.n_clusters = n_clusters
        self.kmeans = None
        self.labels = None
        self.centers = None

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit clustering model.

        Args:
            embeddings: Image embeddings (N, D)

        Returns:
            Cluster labels (N,)
        """
        # Standardize embeddings
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)

        # Fit K-means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        self.labels = self.kmeans.fit_predict(scaled_embeddings)
        self.centers = self.kmeans.cluster_centers_

        logger.info(f"Fitted {self.n_clusters} clusters")
        return self.labels

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new embeddings.

        Args:
            embeddings: Image embeddings (N, D)

        Returns:
            Cluster labels (N,)
        """
        if self.kmeans is None:
            raise RuntimeError("Clusterer not fitted yet. Call fit() first.")

        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        return self.kmeans.predict(scaled_embeddings)

    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        if self.centers is None:
            raise RuntimeError("Clusterer not fitted yet.")
        return self.centers

    def save(self, path: Path) -> None:
        """Save clusterer to disk."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'labels': self.labels,
                'centers': self.centers,
                'n_clusters': self.n_clusters
            }, f)

        logger.info(f"Clusterer saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "ImageClusterer":
        """Load clusterer from disk."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)

        obj = cls(n_clusters=data['n_clusters'])
        obj.kmeans = data['kmeans']
        obj.labels = data['labels']
        obj.centers = data['centers']

        logger.info(f"Clusterer loaded from {path}")
        return obj


class ConditionBasedGrouping:
    """Group images by skin condition."""

    def __init__(self, metadata_df: pd.DataFrame):
        """
        Initialize grouping.

        Args:
            metadata_df: DataFrame with condition labels
        """
        self.metadata_df = metadata_df
        self.condition_groups = self._create_groups()

    def _create_groups(self) -> Dict[str, List[int]]:
        """Create groups of indices by condition."""
        groups = {}
        for idx, row in self.metadata_df.iterrows():
            condition = row.get('condition', 'unknown')
            if condition not in groups:
                groups[condition] = []
            groups[condition].append(idx)
        return groups

    def get_group(self, condition: str) -> List[int]:
        """Get indices for a specific condition."""
        return self.condition_groups.get(condition, [])

    def get_conditions(self) -> List[str]:
        """Get list of all conditions."""
        return list(self.condition_groups.keys())

    def get_group_sizes(self) -> Dict[str, int]:
        """Get size of each group."""
        return {cond: len(indices) for cond, indices in self.condition_groups.items()}
