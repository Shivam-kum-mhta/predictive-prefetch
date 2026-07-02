# The main embeddings interface that the other modules interact with
import chromadb
import os
import numpy as np
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

class NewsEmbeddings:
    def __init__(self, db_path=None):
        # Initialize ChromaDB client
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
        
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Initialize Ollama embedding function
        self.ollama_ef = OllamaEmbeddingFunction(
            url="http://localhost:11434",
            model_name="nomic-embed-text:v1.5",
        )

        self.title_collection_name = "title_embeddings"
        self.abstract_collection_name = "abstract_embeddings"

        # Create or get collections with embedding function
        try:
            self.title_collection = self.client.get_collection(
                name=self.title_collection_name,
                embedding_function=self.ollama_ef
            )
        except:
            self.title_collection = self.client.create_collection(
                name=self.title_collection_name,
                embedding_function=self.ollama_ef,
                metadata={"description": "News article title embeddings"}
            )

        try:
            self.abstract_collection = self.client.get_collection(
                name=self.abstract_collection_name,
                embedding_function=self.ollama_ef
            )
        except:
            self.abstract_collection = self.client.create_collection(
                name=self.abstract_collection_name,
                embedding_function=self.ollama_ef,
                metadata={"description": "News article abstract embeddings"}
            )

    def EmbedTitle(self, titles, ids, metadatas=None):
        """Add title embeddings to the collection"""
        if isinstance(titles, str):
            titles = [titles]
            ids = [ids]
        
        self.title_collection.add(
            documents=titles,
            ids=ids,
            metadatas=metadatas
        )
        return f"Added {len(titles)} title embeddings"
    
    def EmbedAbstract(self, abstracts, ids, metadatas=None):
        """Add abstract embeddings to the collection"""
        if isinstance(abstracts, str):
            abstracts = [abstracts]
            ids = [ids]
            
        self.abstract_collection.add(
            documents=abstracts,
            ids=ids,
            metadatas=metadatas
        )
        return f"Added {len(abstracts)} abstract embeddings"

    def RemoveTitle(self, titleID):
        """Remove a title embedding by ID"""
        self.title_collection.delete(ids=[titleID])
    
    def RemoveAbstract(self, abstractID):
        """Remove an abstract embedding by ID"""
        self.abstract_collection.delete(ids=[abstractID])
    
    def SimilarityBetweenTitles(self, title1ID, title2ID):
        """Calculate cosine similarity between two title embeddings"""
        # Get embeddings for both titles
        result1 = self.title_collection.get(ids=[title1ID], include=["embeddings"])
        result2 = self.title_collection.get(ids=[title2ID], include=["embeddings"])
        
        if len(result1['embeddings']) == 0 or len(result2['embeddings']) == 0:
            return None
            
        emb1 = np.array(result1['embeddings'][0])
        emb2 = np.array(result2['embeddings'][0])
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def SimilarityBetweenAbstracts(self, abstract1ID, abstract2ID):
        """Calculate cosine similarity between two abstract embeddings"""
        result1 = self.abstract_collection.get(ids=[abstract1ID], include=["embeddings"])
        result2 = self.abstract_collection.get(ids=[abstract2ID], include=["embeddings"])
        
        if len(result1['embeddings']) == 0 or len(result2['embeddings']) == 0:
            return None
            
        emb1 = np.array(result1['embeddings'][0])
        emb2 = np.array(result2['embeddings'][0])
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

    def RetrieveSimilarTitles(self, query_text, k=10):
        """Find similar titles using text query"""
        results = self.title_collection.query(
            query_texts=[query_text],
            n_results=k
        )
        return {
            'ids': results['ids'][0],
            'documents': results['documents'][0],
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0] if results['metadatas'] else None
        }

    def RetrieveSimilarAbstracts(self, query_text, k=10):
        """Find similar abstracts using text query"""
        results = self.abstract_collection.query(
            query_texts=[query_text],
            n_results=k
        )
        return {
            'ids': results['ids'][0],
            'documents': results['documents'][0],
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0] if results['metadatas'] else None
        }
    
    def RetrieveSimilarTitlesById(self, titleID, k=10):
        """Find similar titles using an existing title ID"""
        # Get the title document first
        title_doc = self.title_collection.get(ids=[titleID])
        if not title_doc['documents']:
            return None
            
        # Use the document text to query for similar ones
        query_text = title_doc['documents'][0]
        results = self.title_collection.query(
            query_texts=[query_text],
            n_results=k+1  # +1 because it will include the original
        )
        
        # Filter out the original document
        filtered_results = {
            'ids': [],
            'documents': [],
            'distances': [],
            'metadatas': []
        }
        
        for i, doc_id in enumerate(results['ids'][0]):
            if doc_id != titleID:
                filtered_results['ids'].append(doc_id)
                filtered_results['documents'].append(results['documents'][0][i])
                filtered_results['distances'].append(results['distances'][0][i])
                if results['metadatas'] and results['metadatas'][0]:
                    filtered_results['metadatas'].append(results['metadatas'][0][i])
        
        # Limit to k results
        for key in filtered_results:
            filtered_results[key] = filtered_results[key][:k]
            
        return filtered_results

    def RetrieveSimilarAbstractsById(self, abstractID, k=10):
        """Find similar abstracts using an existing abstract ID"""
        # Get the abstract document first
        abstract_doc = self.abstract_collection.get(ids=[abstractID])
        if not abstract_doc['documents']:
            return None
            
        # Use the document text to query for similar ones
        query_text = abstract_doc['documents'][0]
        results = self.abstract_collection.query(
            query_texts=[query_text],
            n_results=k+1  # +1 because it will include the original
        )
        
        # Filter out the original document
        filtered_results = {
            'ids': [],
            'documents': [],
            'distances': [],
            'metadatas': []
        }
        
        for i, doc_id in enumerate(results['ids'][0]):
            if doc_id != abstractID:
                filtered_results['ids'].append(doc_id)
                filtered_results['documents'].append(results['documents'][0][i])
                filtered_results['distances'].append(results['distances'][0][i])
                if results['metadatas'] and results['metadatas'][0]:
                    filtered_results['metadatas'].append(results['metadatas'][0][i])
        
        # Limit to k results
        for key in filtered_results:
            filtered_results[key] = filtered_results[key][:k]
            
        return filtered_results
    
    def GetCollectionStats(self):
        """Get statistics about the collections"""
        return {
            'title_count': self.title_collection.count(),
            'abstract_count': self.abstract_collection.count(),
            'title_collection_name': self.title_collection_name,
            'abstract_collection_name': self.abstract_collection_name
        }