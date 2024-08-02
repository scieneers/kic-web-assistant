import logging

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

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

        logging.basicConfig(
            level="DEBUG" if env.DEBUG_MODE else "INFO",
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
        )
        self.logger = logging.getLogger("loader")

    def extract(
        self,
    ):
        self.logger.info("Loading HPI data from Moochup API...")
        hpi_courses = Moochup(env.DATA_SOURCE_MOOCHUP_HPI_URL).get_course_documents()
        self.logger.info("Finished loading data from Moochup API.")
        self.logger.info("Loading Moodle data from Moochup API...")
        moodle_moochup_courses = Moochup(env.DATA_SOURCE_MOOCHUP_MOODLE_URL).get_course_documents()
        self.logger.info("Finished loading data from Moochup API.")
        self.logger.info("Loading Moodle data from Moodle API...")
        moodle_courses = Moodle().extract()
        self.logger.info("Finished loading data from Moodle API.")
        self.logger.info("Loading Drupal data from Drupal API...")
        drupal_content = Drupal(base_url=env.DRUPAL_URL).extract()

        all_docs = hpi_courses + moodle_moochup_courses + moodle_courses + drupal_content
        vector_store = VectorDBQdrant(version="remote")

        self.logger.debug("Deleting old collection from Qdrant...")
        vector_store.client.delete_collection(collection_name="web_assistant")

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=256, chunk_overlap=16),
                self.embedder,
            ],
            vector_store=vector_store.as_llama_vector_store(collection_name="web_assistant"),
        )

        self.logger.info("Loading Docs into Qdrant...")
        pipeline.run(documents=all_docs)
        self.logger.info("Finished loading Docs into Qdrant.")


if __name__ == "__main__":
    Fetch_Data().extract()
