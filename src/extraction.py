from src.loaders.moochup import fetch_data
from src.loaders.payload import Document

documents = []

# use all loaders 
course_info = fetch_data()
documents.extend(Document(course_info))


# create embeddings

# store in qdrant