import chromadb
from chromadb.utils import embedding_functions
import os

class VectorDB:
    def __init__(self, db_path="data/chroma_db", collection_name="educational_content"):
        self.client = chromadb.PersistentClient(path=db_path)
        # Using a reliable, small open-source embedding model
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_documents(self, documents, ids, metadatas=None):
        """Add documents to the vector database."""
        self.collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Added {len(documents)} documents to Vector DB.")

    def query(self, query_text, n_results=3):
        """Query the vector database for relevant context."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []

# Global instance
vector_db = VectorDB()
