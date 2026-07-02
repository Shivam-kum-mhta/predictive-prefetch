import ollama
import chromadb
import os
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

## testing vector embeddings with ollama and chromadb
## high dimensional vectors for text
# tenant -> database -> collection -> documents -> embeddings

def test_ollama_chromadb():
    print("🚀 Starting Ollama + ChromaDB test...")
    
    # Set up the model and embedding function
    model_name = "nomic-embed-text:v1.5"
    print(f"📝 Using model: {model_name}")
    
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name=model_name,
    )
    
    # Create persistent ChromaDB client in the embeddings folder
    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    print(f"💾 ChromaDB path: {db_path}")
    
    # Create or get collection
    collection_name = "test_ollama_embeddings"
    try:
        # Try to get existing collection first
        collection = client.get_collection(
            name=collection_name,
            embedding_function=ollama_ef
        )
        print(f"📂 Found existing collection: {collection_name}")
    except:
        # Create new collection if it doesn't exist
        collection = client.create_collection(
            name=collection_name,
            embedding_function=ollama_ef,
            metadata={"description": "Test collection for Ollama embeddings"}
        )
        print(f"🆕 Created new collection: {collection_name}")
    
    # Test documents
    test_docs = [
        "This is my first text to embed about technology and AI",
        "This is my second document discussing machine learning models",
        "ChromaDB is a vector database for AI applications",
        "Ollama provides local language model inference",
        "Vector embeddings represent text as high-dimensional vectors"
    ]
    
    # Add documents to collection
    print(f"📄 Adding {len(test_docs)} documents to collection...")
    collection.add(
        documents=test_docs,
        ids=[f"doc_{i}" for i in range(len(test_docs))],
        metadatas=[{"source": f"test_doc_{i}", "type": "test"} for i in range(len(test_docs))]
    )
    
    # Query the collection
    print("\n🔍 Testing queries...")
    query_text = "AI and machine learning"
    results = collection.query(
        query_texts=[query_text],
        n_results=3
    )
    
    print(f"\nQuery: '{query_text}'")
    print("Top 3 similar documents:")
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        print(f"  {i+1}. (distance: {distance:.4f}) {doc}")
    
    # Show collection info
    collection_count = collection.count()
    print(f"\n📊 Collection info:")
    print(f"  - Name: {collection.name}")
    print(f"  - Total documents: {collection_count}")
    print(f"  - Database path: {db_path}")
    
    # Test another query
    query_text2 = "vector database"
    results2 = collection.query(
        query_texts=[query_text2],
        n_results=2
    )
    
    print(f"\nQuery: '{query_text2}'")
    print("Top 2 similar documents:")
    for i, (doc, distance) in enumerate(zip(results2['documents'][0], results2['distances'][0])):
        print(f"  {i+1}. (distance: {distance:.4f}) {doc}")
    
    print("\n✅ Test completed successfully!")
    return collection, client

if __name__ == "__main__":
    try:
        collection, client = test_ollama_chromadb()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Ollama is running on localhost:11434")

