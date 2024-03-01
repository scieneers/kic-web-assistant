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
        # Azure Search
        self.AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
        self.AZURE_OPENAI_URL = os.getenv("AZURE_OPENAI_URL", "")
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
