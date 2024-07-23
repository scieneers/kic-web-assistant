<<<<<<< HEAD
import json

from llama_index.core import Document
=======
>>>>>>> ff295b9 (loader writes module_id as a metadate to vector db)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from src.env import env
from src.llm.LLMs import LLM
from src.loaders.drupal import Drupal
from src.loaders.moochup import Moochup
from src.loaders.moodle import Moodle
from src.vectordb.qdrant import VectorDBQdrant


# A full run takes about 2 hours and 8 minutes (2024-07-16)
class Fetch_Data:
    def __init__(self):
        self.DATA_PATH = "./data"
        self.embedder = LLM().get_embedder()

    def extract(
        self,
    ):
        hpi_courses = Moochup(self.secrets.DATA_SOURCE_MOOCHUP_HPI_URL).get_course_documents()
        moodle_courses1 = Moochup(self.secrets.DATA_SOURCE_MOOCHUP_MOODLE_URL).get_course_documents()
        moodle_docs = Moodle(environment=self.secrets.ENVIRONMENT).extract()
        drupal_courses = Drupal(base_url=self.secrets.DRUPAL_URL).extract()

        all_docs = hpi_courses + moodle_courses1 + moodle_docs + drupal_courses

        vector_store = VectorDBQdrant(version="remote")
        vector_store.client.delete_collection(collection_name="web_assistant")

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=256, chunk_overlap=16),
                self.embedder,
            ],
            vector_store=vector_store.as_llama_vector_store(collection_name="web_assistant"),
        )

        pipeline.run(documents=all_docs)

        return all_docs


if __name__ == "__main__":
    docs = Fetch_Data().extract()
