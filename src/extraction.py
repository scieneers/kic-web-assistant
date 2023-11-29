from src.loaders.moochup import get_course_payloads
from src.loaders.payload import Payload
from src.embedder.multilingual_e5_large import MultilingualE5LargeEmbedder
from src.vectordb.qdrant import VectorDBQdrant

# use all loaders 
documents:list[Payload] = []
course_payloads = get_course_payloads()
documents.extend(course_payloads)

# create embeddings
embedder = MultilingualE5LargeEmbedder()
embeddings = [embedder.embed(payload.vector_content, type="passage") for payload in documents]

# store in db
points = [{'id': i, 'vector': embedding, 'payload': payload.model_dump()} 
          for i, (embedding, payload) in enumerate(zip(embeddings, documents))]
print(f"Example point: {points[0]}")

vectordb = VectorDBQdrant(collection_name='test_collection', vector_size=len(embeddings[0]))
vectordb.upsert(points)

# test
while True:
    query = input("Enter query: ")
    query_embedding = embedder.embed(query)
    search_result = vectordb.search(query_vector=query_embedding, with_payload=True)
    print(search_result)
