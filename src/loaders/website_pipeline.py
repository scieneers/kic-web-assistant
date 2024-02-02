from llama_index import Document
from src.embedder.multilingual_e5_large import MultilingualE5LargeEmbedder
from src.vectordb.qdrant import VectorDBQdrant
from llama_index.text_splitter import SentenceSplitter
from llama_index.ingestion import IngestionPipeline
from src.loaders.moochup import Moochup
from src.env import EnvHelper

# Overview of courses (moochup), website contents (drupal) and support (zammad helpdesk)
env_settings = EnvHelper()

# data extraction 
documents:list[Document] = []
hpi_courses = Moochup(env_settings.DATA_SOURCE_MOOCHUP_HPI_URL).get_course_documents()
documents.extend(hpi_courses)
moodle_courses = Moochup(env_settings.DATA_SOURCE_MOOCHUP_MOODLE_URL).get_course_documents()
documents.extend(moodle_courses)

if env_settings.DEBUG:
    # save documents to local file
    with open('outputs/course_documents.json', 'w') as file:
        file.write('\n'.join([doc.json() for doc in documents]))

# loading pipeline
vector_store = VectorDBQdrant(version='memory')
vector_store.client.delete_collection(collection_name='website')
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=256, chunk_overlap=16),
        MultilingualE5LargeEmbedder(),
    ],
    vector_store=vector_store.as_llama_vector_store(collection_name='website'),
)

pipeline.run(documents=documents)

#print(vector_store.client.scroll(collection_name='assistant'))
print('Completed ingestion pipeline.')
