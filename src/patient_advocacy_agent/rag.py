"""RAG (Retrieval-Augmented Generation) system for medical knowledge and case retrieval."""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel

# LangChain imports (compatible with langchain>=0.1.0)
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.schema import Document
    except ImportError:
        # Fallback: create simple Document class if LangChain not fully installed
        class Document:
            def __init__(self, page_content: str, metadata: Optional[Dict] = None):
                self.page_content = page_content
                self.metadata = metadata or {}

logger = logging.getLogger(__name__)


class RetrievedCase(BaseModel):
    """Retrieved case from knowledge base."""
    case_id: str
    condition: str
    similarity_score: float
    symptoms: List[str]
    notes: Optional[str] = None
    severity: Optional[str] = None
    image_path: Optional[str] = None


class MedicalKnowledgeBase:
    """Knowledge base for medical information about skin conditions."""

    def __init__(
        self,
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize medical knowledge base.

        Args:
            embeddings_model: HuggingFace model for text embeddings
        """
        self.embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.vector_store = None
        self.documents = []

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to knowledge base.

        Args:
            documents: List of LangChain Document objects
        """
        self.documents.extend(documents)

        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                documents,
                self.embeddings_model
            )
        else:
            # Add to existing vector store
            self.vector_store.add_documents(documents)

        logger.info(f"Added {len(documents)} documents to knowledge base")

    def add_from_dataframe(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
        metadata_columns: Optional[List[str]] = None
    ) -> None:
        """
        Add documents from DataFrame.

        Args:
            df: DataFrame with document content
            text_columns: Columns to combine as document text
            metadata_columns: Columns to include as metadata
        """
        documents = []

        for idx, row in df.iterrows():
            # Combine text columns
            text = " ".join([str(row[col]) for col in text_columns if col in row])

            # Extract metadata
            metadata = {}
            if metadata_columns:
                metadata = {col: row[col] for col in metadata_columns if col in row}

            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

        self.add_documents(documents)

    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base for relevant documents.

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of relevant documents with scores
        """
        if self.vector_store is None:
            logger.warning("Knowledge base is empty")
            return []

        results = self.vector_store.similarity_search_with_score(query, k=k)

        retrieved = []
        for doc, score in results:
            if score_threshold and score < score_threshold:
                continue

            retrieved.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            })

        return retrieved

    def save(self, path: Path) -> None:
        """Save knowledge base to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.vector_store:
            self.vector_store.save_local(str(path / "vector_store"))

        logger.info(f"Knowledge base saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "MedicalKnowledgeBase":
        """Load knowledge base from disk."""
        path = Path(path)

        obj = cls()

        if (path / "vector_store").exists():
            obj.vector_store = FAISS.load_local(
                str(path / "vector_store"),
                obj.embeddings_model
            )

        logger.info(f"Knowledge base loaded from {path}")
        return obj


class CaseRetriever:
    """Retrieve similar cases from the database."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        embeddings: np.ndarray,
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize case retriever.

        Args:
            metadata_df: DataFrame with case metadata
            embeddings: Pre-computed embeddings for cases
            embeddings_model: HuggingFace model for text embeddings
        """
        self.metadata_df = metadata_df
        self.embeddings = embeddings
        self.text_embeddings_model = HuggingFaceEmbeddings(
            model_name=embeddings_model
        )

        # Build vector store from case descriptions
        self._build_vector_store()

    def _build_vector_store(self) -> None:
        """Build vector store from case descriptions."""
        documents = []

        for idx, row in self.metadata_df.iterrows():
            # Combine available text fields
            text_parts = []

            if 'condition' in row:
                text_parts.append(f"Condition: {row['condition']}")

            if 'symptoms' in row and row['symptoms']:
                symptoms = row['symptoms']
                if isinstance(symptoms, str):
                    symptoms = symptoms.split(',')
                text_parts.append(f"Symptoms: {', '.join(symptoms)}")

            if 'description' in row:
                text_parts.append(f"Description: {row['description']}")

            if 'notes' in row and row['notes']:
                text_parts.append(f"Notes: {row['notes']}")

            text = " ".join(text_parts)

            metadata = {
                'case_id': row.get('image_id', f'case_{idx}'),
                'condition': row.get('condition', 'unknown'),
                'index': idx
            }

            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

        if documents:
            self.vector_store = FAISS.from_documents(
                documents,
                self.text_embeddings_model
            )
            logger.info(f"Built vector store with {len(documents)} cases")
        else:
            self.vector_store = None
            logger.warning("No documents to build vector store")

    def retrieve_similar_cases(
        self,
        condition: str,
        symptoms: Optional[List[str]] = None,
        k: int = 5
    ) -> List[RetrievedCase]:
        """
        Retrieve cases similar to query.

        Args:
            condition: Target skin condition
            symptoms: List of symptoms
            k: Number of cases to retrieve

        Returns:
            List of retrieved cases
        """
        if self.vector_store is None:
            return []

        # Build query from condition and symptoms
        query_parts = [f"Condition: {condition}"]
        if symptoms:
            query_parts.append(f"Symptoms: {', '.join(symptoms)}")

        query = " ".join(query_parts)

        # Search vector store
        results = self.vector_store.similarity_search_with_score(query, k=k)

        retrieved_cases = []
        for doc, score in results:
            case_idx = doc.metadata.get('index', 0)
            row = self.metadata_df.iloc[case_idx]

            case = RetrievedCase(
                case_id=doc.metadata.get('case_id', f'case_{case_idx}'),
                condition=doc.metadata.get('condition', 'unknown'),
                similarity_score=float(score),
                symptoms=row.get('symptoms', []),
                notes=row.get('notes'),
                severity=row.get('severity'),
                image_path=row.get('image_path')
            )
            retrieved_cases.append(case)

        return retrieved_cases

    def retrieve_by_embedding_similarity(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_condition: Optional[str] = None
    ) -> List[RetrievedCase]:
        """
        Retrieve cases by image embedding similarity.

        Args:
            query_embedding: Query image embedding
            k: Number of cases to retrieve
            filter_condition: Optional condition to filter by

        Returns:
            List of retrieved cases
        """
        # Compute similarity scores
        similarities = np.dot(self.embeddings, query_embedding)

        if filter_condition:
            mask = self.metadata_df['condition'] == filter_condition
            similarities = similarities * mask.values.astype(float)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]

        retrieved_cases = []
        for idx in top_indices:
            if similarities[idx] <= 0 and filter_condition:
                continue

            row = self.metadata_df.iloc[idx]
            case = RetrievedCase(
                case_id=row.get('image_id', f'case_{idx}'),
                condition=row.get('condition', 'unknown'),
                similarity_score=float(similarities[idx]),
                symptoms=row.get('symptoms', []),
                notes=row.get('notes'),
                severity=row.get('severity'),
                image_path=row.get('image_path')
            )
            retrieved_cases.append(case)

        return retrieved_cases

    def save(self, path: Path) -> None:
        """Save retriever to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.vector_store:
            self.vector_store.save_local(str(path / "case_vector_store"))

        # Save metadata and embeddings
        self.metadata_df.to_csv(path / "metadata.csv", index=False)
        np.save(path / "embeddings.npy", self.embeddings)

        logger.info(f"Case retriever saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "CaseRetriever":
        """Load retriever from disk."""
        path = Path(path)

        # Load metadata and embeddings
        metadata_df = pd.read_csv(path / "metadata.csv")
        embeddings = np.load(path / "embeddings.npy")

        obj = cls(metadata_df, embeddings)

        # Load vector store if it exists
        if (path / "case_vector_store").exists():
            obj.vector_store = FAISS.load_local(
                str(path / "case_vector_store"),
                obj.text_embeddings_model
            )

        logger.info(f"Case retriever loaded from {path}")
        return obj


class RAGPipeline:
    """End-to-end RAG pipeline for retrieving relevant medical information."""

    def __init__(
        self,
        case_retriever: CaseRetriever,
        knowledge_base: MedicalKnowledgeBase
    ):
        """
        Initialize RAG pipeline.

        Args:
            case_retriever: CaseRetriever instance
            knowledge_base: MedicalKnowledgeBase instance
        """
        self.case_retriever = case_retriever
        self.knowledge_base = knowledge_base

    def retrieve_context(
        self,
        condition: str,
        symptoms: Optional[List[str]] = None,
        query_embedding: Optional[np.ndarray] = None,
        num_cases: int = 5,
        num_knowledge: int = 3
    ) -> Dict[str, Any]:
        """
        Retrieve comprehensive context for a patient case.

        Args:
            condition: Target skin condition
            symptoms: List of symptoms
            query_embedding: Optional embedding for similarity search
            num_cases: Number of similar cases to retrieve
            num_knowledge: Number of knowledge documents to retrieve

        Returns:
            Dictionary with retrieved context
        """
        context = {
            'condition': condition,
            'symptoms': symptoms or [],
            'similar_cases': [],
            'knowledge_docs': []
        }

        # Retrieve similar cases
        if query_embedding is not None:
            cases = self.case_retriever.retrieve_by_embedding_similarity(
                query_embedding,
                k=num_cases,
                filter_condition=condition
            )
        else:
            cases = self.case_retriever.retrieve_similar_cases(
                condition,
                symptoms,
                k=num_cases
            )

        context['similar_cases'] = [case.model_dump() for case in cases]

        # Retrieve knowledge documents
        query = f"{condition}"
        if symptoms:
            query += f" with {', '.join(symptoms)}"

        knowledge_docs = self.knowledge_base.search(query, k=num_knowledge)
        context['knowledge_docs'] = knowledge_docs

        return context

    def save(self, path: Path) -> None:
        """Save RAG pipeline to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.case_retriever.save(path / "case_retriever")
        self.knowledge_base.save(path / "knowledge_base")

        logger.info(f"RAG pipeline saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "RAGPipeline":
        """Load RAG pipeline from disk."""
        path = Path(path)

        case_retriever = CaseRetriever.load(path / "case_retriever")
        knowledge_base = MedicalKnowledgeBase.load(path / "knowledge_base")

        logger.info(f"RAG pipeline loaded from {path}")
        return cls(case_retriever, knowledge_base)
