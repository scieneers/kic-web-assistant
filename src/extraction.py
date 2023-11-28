from src.loaders.moochup import get_course_payloads
from src.loaders.payload import Payload
from src.embedder.multilingual_e5_large import MultilingualE5LargeEmbedder


# use all loaders 
documents:list[Payload] = []
course_payloads = get_course_payloads()
documents.extend(course_payloads)

# create embeddings
embedder = MultilingualE5LargeEmbedder()
embeddings = [embedder.embed(payload.vector_content) for payload in documents]

# store in qdrant
qdrant_points = [{'id': i, 'vector': embedding, 'payload': payload.model_dump()} 
                 for i, (embedding, payload) in enumerate(zip(embeddings, documents))]

print(qdrant_points)