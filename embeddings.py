"""
Embedding Generation Module using Sentence Transformers
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
import config


class EmbeddingGenerator:
    """
    Generate embeddings using Sentence Transformers.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        Load the embedding model (lazy loading).
        """
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Embedding model loaded successfully!")
    
    def generate(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return None
    
    def generate_single(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        """
        return self.model.get_sentence_embedding_dimension()


# Create a default instance
embedding_generator = EmbeddingGenerator()