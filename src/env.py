import logging
import os

from dotenv import find_dotenv, load_dotenv


class EnvHelper:
    def __init__(self, production: bool = False) -> None:
        """Helper class for environment variables. Loads production variables if production=True, or env variable 'ENVIRONMENT' is set to 'PRODUCTION'"""
        if not find_dotenv():
            logging.warning(
                "No .env file found, using environment variables. Use task decrypt-env to generate a .env file."
            )
        load_dotenv()
        # Environment setup
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "STAGING")

        # Azure / GPT-4
        self.AZURE_OPENAI_GPT4_KEY = os.getenv("AZURE_OPENAI_GPT4_KEY", "")
        self.AZURE_OPENAI_GPT4_URL = os.getenv("AZURE_OPENAI_GPT4_URL", "")
        self.AZURE_OPENAI_GPT4_API_VERSION = os.getenv("AZURE_OPENAI_GPT4_API_VERSION", "")
        self.AZURE_OPENAI_GPT4_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "")
        self.AZURE_OPENAI_GPT4_MODEL = os.getenv("AZURE_OPENAI_GPT4_MODEL", "")

        # Azure / Mistral
        self.AZURE_OPENAI_MISTRAL_URL = os.getenv("AZURE_OPENAI_MISTRAL_URL", "")
        self.AZURE_OPENAI_MISTRAL_KEY = os.getenv("AZURE_OPENAI_MISTRAL_KEY", "")

        # Azure / Llama3
        self.AZURE_OPENAI_LLAMA3_URL = os.getenv("AZURE_OPENAI_LLAMA3_URL", "")
        self.AZURE_OPENAI_LLAMA3_KEY = os.getenv("AZURE_OPENAI_LLAMA3_KEY", "")

        # Aleph Alpha
        self.AA_TOKEN = os.getenv("AA_TOKEN", "")

        # Azure Embedder
        self.AZURE_OPENAI_EMBEDDER_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDER_API_VERSION", "")
        self.AZURE_OPENAI_EMBEDDER_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDER_API_KEY", "")
        self.AZURE_OPENAI_EMBEDDER_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDER_ENDPOINT", "")
        self.AZURE_OPENAI_EMBEDDER_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDER_DEPLOYMENT", "")
        self.AZURE_OPENAI_EMBEDDER_MODEL = os.getenv("AZURE_OPENAI_EMBEDDER_MODEL", "")

        # Langfuse
        self.LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "")
        self.LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")

        # Drupal
        self.DRUPAL_URL = os.getenv("DRUPAL_URL", "")
        self.DRUPAL_CLIENT_ID = os.getenv("DRUPAL_CLIENT_ID", "")
        self.DRUPAL_CLIENT_SECRET = os.getenv("DRUPAL_CLIENT_SECRET", "")
        self.DRUPAL_USERNAME = os.getenv("DRUPAL_USERNAME", "")
        self.DRUPAL_PASSWORD = os.getenv("DRUPAL_PASSWORD", "")
        self.DRUPAL_GRANT_TYPE = os.getenv("DRUPAL_GRANT_TYPE", "")

        # Vector db
        self.QDRANT_TOKEN = os.getenv("QDRANT_TOKEN", "")
        self.QDRANT_URL = os.getenv("QDRANT_URL", "")
        # Data sources
        if production or self.ENVIRONMENT == "PRODUCTION":
            self.DATA_SOURCE_MOODLE_URL = os.getenv("DATA_SOURCE_PRODUCTION_MOODLE_URL", "")
            self.DATA_SOURCE_MOODLE_TOKEN = os.getenv("DATA_SOURCE_PRODUCTION_MOODLE_TOKEN", "")
        else:
            self.DATA_SOURCE_MOODLE_URL = os.getenv("DATA_SOURCE_STAGING_MOODLE_URL", "")
            self.DATA_SOURCE_MOODLE_TOKEN = os.getenv("DATA_SOURCE_STAGING_MOODLE_TOKEN", "")
        self.DATA_SOURCE_MOOCHUP_HPI_URL = os.getenv("DATA_SOURCE_MOOCHUP_HPI_URL", "")
        self.DATA_SOURCE_MOOCHUP_MOODLE_URL = os.getenv("DATA_SOURCE_MOOCHUP_MOODLE_URL", "")
        # Vimeo
        self.VIMEO_PAT = os.getenv("VIMEO_PAT", "")
        # Debug Setting
        self.DEBUG = bool(os.getenv("DEBUG", "False"))

    @staticmethod
    def check_env():
        for attr, value in EnvHelper().__dict__.items():
            if value == "":
                logging.warning(f"{attr} is not set in the environment variables.")

    def __getattribute__(self, name):
        """Since not all environment variables are always required or set, this will raise an error if the attribute is not set during runtime."""
        value = object.__getattribute__(self, name)
        if value == "":
            raise AttributeError(f"{name} is requested but not set in the environment variables")
        return value


if __name__ == "__main__":
    EnvHelper.check_env()
