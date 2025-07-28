"""
Stage 4: ChromaDB Storage for documents, embeddings, and metadata
"""

import logging
from typing import List, Dict, Any, Optional, Union
import uuid
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-based vector store for RAG applications."""

    def __init__(self, collection_name: str = "html_documents", persist_directory: str = "./chroma_db"):
        """
        Initialize the ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None

        # Initialize ChromaDB client
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            logger.info(f"Initializing ChromaDB client with persist directory: {self.persist_directory}")

            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Create or get collection
            # Note: We'll handle embeddings manually since we're using our own embedding model
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "HTML document collection for RAG pipeline",
                    "hnsw:space": "cosine"
                }
            )

            logger.info(f"ChromaDB collection '{self.collection_name}' initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {e}")
            raise

    def add_documents(self, embedded_blocks: List[Dict[str, Any]]) -> None:
        """
        Add documents with embeddings and metadata to ChromaDB.

        Args:
            embedded_blocks: List of text blocks with embeddings from Stage 3
        """
        try:
            logger.info(f"Adding {len(embedded_blocks)} documents to ChromaDB")

            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for block in embedded_blocks:
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)

                # Extract embedding (convert to list for ChromaDB)
                embedding = block['embedding']
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                embeddings.append(embedding)

                # Document text
                documents.append(block['text'])

                # Metadata (exclude embedding from metadata)
                metadata = {k: v for k, v in block.items() if k not in ['text', 'embedding']}

                # Ensure all metadata values are JSON serializable
                for key, value in metadata.items():
                    if isinstance(value, np.ndarray):
                        metadata[key] = value.tolist()
                    elif value is None:
                        metadata[key] = ""
                    else:
                        metadata[key] = str(value)

                metadatas.append(metadata)

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"Successfully added {len(embedded_blocks)} documents to ChromaDB")

        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

    def search_by_similarity(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic similarity.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            metadata_filter: Optional metadata filter

        Returns:
            List of similar documents with metadata and similarity scores
        """
        try:
            logger.info(f"Searching for {n_results} similar documents")

            # Convert embedding to list
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=metadata_filter,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    result = {
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                    }
                    formatted_results.append(result)

            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def search_by_text(
        self,
        query_text: str,
        n_results: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents using text query (ChromaDB will handle embedding).

        Args:
            query_text: Query text
            n_results: Number of results to return
            metadata_filter: Optional metadata filter

        Returns:
            List of similar documents with metadata and similarity scores
        """
        try:
            logger.info(f"Searching with text query: '{query_text[:50]}...'")

            # Perform text search
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=metadata_filter,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    result = {
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity_score': 1 - results['distances'][0][i]
                    }
                    formatted_results.append(result)

            logger.info(f"Found {len(formatted_results)} documents for text query")
            return formatted_results

        except Exception as e:
            logger.error(f"Error during text search: {e}")
            return []

    def filter_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter documents by metadata criteria.

        Args:
            metadata_filter: Metadata filter criteria
            limit: Maximum number of results to return

        Returns:
            List of documents matching the filter criteria
        """
        try:
            logger.info(f"Filtering documents by metadata: {metadata_filter}")

            # Get all documents matching the filter
            results = self.collection.get(
                where=metadata_filter,
                limit=limit,
                include=['documents', 'metadatas']
            )

            # Format results
            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    result = {
                        'text': results['documents'][i],
                        'metadata': results['metadatas'][i]
                    }
                    formatted_results.append(result)

            logger.info(f"Found {len(formatted_results)} documents matching filter")
            return formatted_results

        except Exception as e:
            logger.error(f"Error filtering by metadata: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_directory': self.persist_directory
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted successfully")

        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

    def reset_collection(self) -> None:
        """Reset the collection (delete and recreate)."""
        try:
            # Delete existing collection
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass  # Collection might not exist

            # Recreate collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "HTML document collection for RAG pipeline"}
            )

            logger.info(f"Collection '{self.collection_name}' reset successfully")

        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

    def update_document(self, doc_id: str, new_text: str, new_metadata: Dict[str, Any], new_embedding: np.ndarray) -> None:
        """
        Update an existing document.

        Args:
            doc_id: Document ID to update
            new_text: New document text
            new_metadata: New metadata
            new_embedding: New embedding vector
        """
        try:
            # Convert embedding to list
            if isinstance(new_embedding, np.ndarray):
                new_embedding = new_embedding.tolist()

            # Ensure metadata is JSON serializable
            for key, value in new_metadata.items():
                if isinstance(value, np.ndarray):
                    new_metadata[key] = value.tolist()
                elif value is None:
                    new_metadata[key] = ""
                else:
                    new_metadata[key] = str(value)

            # Update document
            self.collection.update(
                ids=[doc_id],
                embeddings=[new_embedding],
                documents=[new_text],
                metadatas=[new_metadata]
            )

            logger.info(f"Document {doc_id} updated successfully")

        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Vector store cleanup completed")