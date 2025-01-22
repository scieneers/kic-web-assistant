import logging
import azure.functions as func

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import models

from src.env import env
from src.llm.LLMs import LLM
from src.loaders.drupal import Drupal
from src.loaders.moochup import Moochup
from src.loaders.moodle import Moodle
from src.vectordb.qdrant import VectorDBQdrant

DEFAULT_COLLECTION = "web_assistant"


# A full run takes about 3 hours (2024-08-02)
class Fetch_Data:
    def sanity_check(self):
        # Check if URLs are missing in metadata,
        # every point needs a non-empty url field in the metadata
        query_filter = models.Filter(must=[models.IsEmptyCondition(is_empty=models.PayloadField(key="url"))])

        if self.vector_store.query_with_filter(DEFAULT_COLLECTION, query_filter) != ([], None):
            self.logger.error("Missing URLs in Metadata, linking to content not possible in all cases")

    def __init__(self):
        self.DATA_PATH = "./data"
        self.embedder = LLM().get_embedder()
        self.logger = logging.getLogger("loader")
        self.logger.propagate = False
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "{asctime} - {levelname:<8} - {message}",
            style="{",
            datefmt="%d-%b-%y %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG if env.DEBUG_MODE else logging.INFO)
        self.vector_store = VectorDBQdrant(version="remote")

        self.logger.info("Starting data extraction...")

    def extract(
        self,
    ):
        self.logger.info("Loading HPI data from Moochup API...")
        hpi_courses = Moochup(env.DATA_SOURCE_MOOCHUP_HPI_URL).get_course_documents()
        self.logger.info("Finished loading HPI data from Moochup API.")
        self.logger.info("Loading Moodle data from Moochup API...")
        moodle_moochup_courses = Moochup(env.DATA_SOURCE_MOOCHUP_MOODLE_URL).get_course_documents()
        self.logger.info("Finished loading data from Moochup API.")
        self.logger.info("Loading Moodle data from Moodle API...")
        moodle_courses = Moodle().extract()
        self.logger.info("Finished loading data from Moodle API.")
        self.logger.info("Loading Drupal data from Drupal API...")
        drupal_content = Drupal(
            base_url=env.DRUPAL_URL,
            username=env.DRUPAL_USERNAME,
            client_id=env.DRUPAL_CLIENT_ID,
            client_secret=env.DRUPAL_CLIENT_SECRET,
            grant_type=env.DRUPAL_GRANT_TYPE,
        ).extract()

        all_docs = hpi_courses + moodle_moochup_courses + moodle_courses + drupal_content

        self.logger.debug("Deleting old collection from Qdrant...")
        self.vector_store.client.delete_collection(collection_name=DEFAULT_COLLECTION)

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=256, chunk_overlap=16),
                self.embedder,
            ],
            vector_store=self.vector_store.as_llama_vector_store(collection_name=DEFAULT_COLLECTION),
        )

        self.logger.info("Loading Docs into Qdrant...")
        pipeline.run(documents=all_docs)
        self.logger.info("Finished loading Docs into Qdrant.")
        self.sanity_check()


if __name__ == "__main__":
    Fetch_Data().extract()

app = func.FunctionApp()

@app.timer_trigger(schedule="0 4 * * 3", run_on_startup=False,
              use_monitor=False) 
def timer_trigger() -> None:
    Fetch_Data().extract()