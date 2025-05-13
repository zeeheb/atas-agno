"""
Persistence module for saving and loading vector store data
"""
import os
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

# Configure logging
logger = logging.getLogger(__name__)

class VectorStorePersistence:
    """
    Handles saving and loading vector store data to/from disk
    """
    
    def __init__(self, app_data_dir: str = "app_data"):
        """
        Initialize the persistence handler
        
        Args:
            app_data_dir: Directory to store application data
        """
        self.app_data_dir = Path(app_data_dir)
        self.vectors_file = self.app_data_dir / "vectors.npy"
        self.documents_file = self.app_data_dir / "documents.json"
        self.metadata_file = self.app_data_dir / "metadata.json"
        self._ensure_dir_exists()
        
    def _ensure_dir_exists(self):
        """Create the app data directory if it doesn't exist"""
        self.app_data_dir.mkdir(exist_ok=True, parents=True)
        
    def save_vector_store(self, vector_store) -> bool:
        """
        Save vector store data to disk
        
        Args:
            vector_store: The SimpleVectorStore instance to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if there's actually data to save
            if not vector_store.documents:
                logger.info("No data to save - vector store is empty")
                return False
                
            # Save vectors using numpy (efficient for numerical data)
            if vector_store.vectors:
                np.save(self.vectors_file, np.array(vector_store.vectors))
            
            # Save documents and metadata as JSON
            documents_data = {
                "documents": vector_store.documents,
                "ids": vector_store.ids
            }
            
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(documents_data, f, ensure_ascii=False, indent=2)
                
            # Save metadata separately (might contain complex data)
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(vector_store.metadatas, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Successfully saved vector store data with {len(vector_store.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
            
    def load_vector_store(self, vector_store) -> bool:
        """
        Load vector store data from disk
        
        Args:
            vector_store: The SimpleVectorStore instance to populate
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if files exist
            if not self.vectors_file.exists() or not self.documents_file.exists() or not self.metadata_file.exists():
                logger.info("No saved vector store data found")
                return False
                
            # Load vectors from numpy file
            vectors = np.load(self.vectors_file, allow_pickle=True)
            
            # Load documents and ids
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)
                
            # Load metadata
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadatas = json.load(f)
                
            # Populate the vector store
            vector_store.vectors = [v.tolist() for v in vectors]
            vector_store.documents = documents_data["documents"]
            vector_store.ids = documents_data["ids"]
            vector_store.metadatas = metadatas
            
            # Verify that all arrays have the same length
            counts = [
                len(vector_store.vectors),
                len(vector_store.documents),
                len(vector_store.ids),
                len(vector_store.metadatas)
            ]
            
            if len(set(counts)) > 1:
                logger.warning(f"Inconsistent data lengths: vectors={counts[0]}, documents={counts[1]}, ids={counts[2]}, metadata={counts[3]}")
                
            logger.info(f"Successfully loaded vector store data with {len(vector_store.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def get_document_count(self) -> int:
        """
        Get the number of saved documents without loading the entire store
        
        Returns:
            int: Number of documents or 0 if no data or error
        """
        try:
            if not self.documents_file.exists():
                return 0
                
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)
                return len(documents_data.get("documents", []))
                
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0 