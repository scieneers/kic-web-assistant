import json

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from src.env import EnvHelper
from src.loaders.drupal import Drupal
from src.loaders.moochup import Moochup
from src.loaders.moodle import Moodle
from src.vectordb.qdrant import VectorDBQdrant


# A full run takes about 2 hours and 8 minutes (2024-07-16)
class Fetch_Data:
    def __init__(self, path=None):
        self.secrets = EnvHelper()

        self.DATA_PATH = "./data"

        self.embedder = AzureOpenAIEmbedding(
            model=self.secrets.AZURE_OPENAI_EMBEDDER_MODEL,
            deployment_name=self.secrets.AZURE_OPENAI_EMBEDDER_DEPLOYMENT,
            api_key=self.secrets.AZURE_OPENAI_EMBEDDER_API_KEY,
            azure_endpoint=self.secrets.AZURE_OPENAI_EMBEDDER_ENDPOINT,
            api_version=self.secrets.AZURE_OPENAI_EMBEDDER_API_VERSION,
        )

    def extract(
        self,
    ):
        hpi_courses = Moochup(self.secrets.DATA_SOURCE_MOOCHUP_HPI_URL).get_course_documents()
        hpi_text_list = [json.loads(doc.json())["text"] for doc in hpi_courses]

        moodle_courses1 = Moochup(self.secrets.DATA_SOURCE_MOOCHUP_MOODLE_URL).get_course_documents()
        moodle_text_list1 = [json.loads(doc.json())["text"] for doc in moodle_courses1]

        moodle_courses2 = Moodle(environment=self.secrets.ENVIRONMENT).extract()
        moodle_text_list2 = [json.loads(doc.json())["text"] for doc in moodle_courses2]

        # drupal_courses = Drupal(base_url=self.secrets.DRUPAL_URL).extract()
        # drupal_text_list = [json.loads(doc.json())["text"] for doc in drupal_courses]

        all_text = hpi_text_list + moodle_text_list1 + moodle_text_list2  # + drupal_text_list

        documents = [Document(text=t) for t in all_text]

        vector_store = VectorDBQdrant(version="remote")
        vector_store.client.delete_collection(collection_name="web_assistant")

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=256, chunk_overlap=16),
                self.embedder,
            ],
            vector_store=vector_store.as_llama_vector_store(collection_name="web_assistant"),
        )

        pipeline.run(documents=documents)

        # file = open(f'{self.DATA_PATH}/course_data.txt', 'w', encoding="utf8")

        # for text in all_text:
        #     file.write(text)
        #     file.write("\n")

        return documents


if __name__ == "__main__":
    docs = Fetch_Data().extract()
