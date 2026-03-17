"""
Vector Store Module using FAISS
"""

from typing import List, Dict, Optional, Tuple
import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from embeddings import embedding_generator


class VectorStore:
    """
    FAISS-based vector store for storing and retrieving document embeddings.
    """
    
    def __init__(self):
        """
        Initialize the vector store.
        """
        self.vector_store = None
        self.documents = []
        self.video_metadata = {}
    
    def create_from_transcript(self, transcript_data: Dict, video_url: str) -> bool:
        """
        Create vector store from transcript data.
        
        Args:
            transcript_data: Dictionary containing transcript chunks
            video_url: Original YouTube URL
            
        Returns:
            True if successful, False otherwise
        """
        try:
            chunks = transcript_data['chunks']
            video_id = transcript_data['video_id']
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                if len(chunk['text'].strip()) > 0:
                    doc = Document(
                        page_content=chunk['text'],
                        metadata={
                            'video_id': video_id,
                            'video_url': video_url,
                            'chunk_id': i,
                            'start_time': chunk['start'],
                            'end_time': chunk['end'],
                            'duration': chunk['duration']
                        }
                    )
                    documents.append(doc)
            
            if not documents:
                return False
            
            # Split documents if they're too long
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            # Create embeddings
            texts = [doc.page_content for doc in split_docs]
            embeddings = embedding_generator.generate(texts)
            
            if not embeddings:
                return False
            
            # Create FAISS index
            dimension = len(embeddings[0])
            index = faiss.IndexFlatL2(dimension)
            
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            index.add(embeddings_array)
            
            # Create docstore
            docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(split_docs)})
            index_to_docstore_id = {i: str(i) for i in range(len(split_docs))}
            
            # Create LangChain FAISS wrapper
            self.vector_store = FAISS(
                embedding_function=embedding_generator.model.encode,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )
            
            self.documents = split_docs
            self.video_metadata = {
                'video_id': video_id,
                'video_url': video_url,
                'total_chunks': len(split_docs),
                'language': transcript_data.get('language', 'en'),
                'total_duration': transcript_data.get('total_duration', 0)
            }
            
            return True
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if self.vector_store is None:
            return []
        
        k = k or config.TOP_K_RESULTS
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with similarity scores.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if self.vector_store is None:
            return []
        
        k = k or config.TOP_K_RESULTS
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def get_context_for_query(self, query: str) -> str:
        """
        Get concatenated context string for a query.
        
        Args:
            query: Query string
            
        Returns:
            Concatenated context from relevant documents
        """
        results = self.similarity_search(query)
        
        if not results:
            return ""
        
        context_parts = []
        for doc in results:
            context_parts.append(doc.page_content)
        
        return "\n\n".join(context_parts)
    
    def clear(self):
        """
        Clear the vector store.
        """
        self.vector_store = None
        self.documents = []
        self.video_metadata = {}


# Create a global instance
vector_store = VectorStore()