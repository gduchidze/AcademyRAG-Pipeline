from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)
try:
    client.get_collections()
    print("Qdrant Connection is OK ")
except Exception as e:
    print(f"Error: {str(e)}")