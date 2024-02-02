from src.loaders.moodle import Moodle
from llama_index import Document
from src.embedder.multilingual_e5_large import MultilingualE5LargeEmbedder
from src.vectordb.qdrant import VectorDBQdrant
from llama_index.text_splitter import SentenceSplitter
from llama_index.ingestion import IngestionPipeline

# Index for chat with your course content

# data extraction 
documents:list[Document] = []
courses = Moodle(environment='PRODUCTION').extract()
documents.extend(courses)


# loading pipeline
vector_store = VectorDBQdrant(version='disk')
vector_store.client.delete_collection(collection_name='assistant')
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=256, chunk_overlap=16),
        MultilingualE5LargeEmbedder(),
    ],
    vector_store=vector_store.as_llama_vector_store(collection_name='assistant'),
)

pipeline.run(documents=documents)

#print(vector_store.client.scroll(collection_name='assistant'))
print('Completed ingestion pipeline.')