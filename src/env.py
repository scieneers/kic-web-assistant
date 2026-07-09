import json
import logging
import os
from typing import Any

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field, field_validator


class EnvHelper(BaseModel):
    """Environment helper. Loads a variable from (1) .env file, (2) environment, (3) key vault, (4) pydantic default.
    If a variable without a default value retrieved, an error is raised."""

    ENVIRONMENT: str = Field(
        default="DEV", description="Whether to use production or dev APIs from ki-campus sites"
    )
    DEBUG_MODE: bool = False
    AUDIO_TRANSCRIPTION_ENABLED: bool = False
    REST_API_KEYS: list[str] = []

    # Base URL for the FastAPI service (used by Streamlit when calling via HTTP)
    REST_API_URL: str = "http://localhost:8000"

    AZURE_OPENAI_URL: str = "UNSET"
    AZURE_OPENAI_API_KEY: str = "UNSET"

    AZURE_FALLBACK_DEPLOYMENT: str = "UNSET"
    AZURE_FALLBACK_MODEL: str = "UNSET"
    AZURE_MINI_DEPLOYMENT: str = "UNSET"
    AZURE_MINI_MODEL: str = "UNSET"
    AZURE_OPENAI_EMBEDDER_DEPLOYMENT: str = "UNSET"
    AZURE_OPENAI_EMBEDDER_MODEL: str = "UNSET"

    #AZURE_MISTRAL_URL: str = "UNSET" <-- Nicht in Azure Key Vault
    AZURE_MISTRAL_KEY: str = "UNSET"

    GWDG_URL: str = "UNSET"
    GWDG_API_KEY: str = "UNSET"
    GWDG_MODEL: str = "UNSET"

    LANGFUSE_HOST: str = "UNSET"
    LANGFUSE_PUBLIC_KEY: str = "UNSET"
    LANGFUSE_SECRET_KEY: str = "UNSET"

    DRUPAL_URL: str = "https://ki-campus.org"
    DRUPAL_CLIENT_ID: str = "UNSET"
    DRUPAL_CLIENT_SECRET: str = "UNSET"
    DRUPAL_USERNAME: str = "UNSET"
    DRUPAL_PASSWORD: str = "UNSET"
    DRUPAL_GRANT_TYPE: str = "password"
    # If True, missing/invalid Drupal credentials abort the ingest run instead of
    # silently falling back to public-only content (which risks stale-deleting
    # protected courses, since the run would then "successfully" see less than
    # what's actually indexed).
    DRUPAL_AUTH_REQUIRED: bool = False

    AZURE_SEARCH_ENDPOINT: str = "UNSET"
    AZURE_SEARCH_INDEX: str = "aichat"

    # Reranker backend for the RAG pipeline: "llm", "azure_semantic" or "bge".
    # "azure_semantic" runs INTEGRATED: the semantic ranker rescores inside the
    # retrieval call itself (one search, no separate rerank round-trip); the
    # rerank node then only cuts to top_n and applies MIN_RERANKER_SCORE.
    # MIN_RERANKER_SCORE is a relevance cutoff on a normalized 0-1 scale;
    # 0.0 disables filtering. If all chunks fall below the cutoff, the
    # assistant answers with the no-answer fallback instead of a weak answer.
    # (Azure reranker_score is 0-4; Microsofts guidance: below ~2.0 is weak →
    # normalized starting point ~0.5, calibrate with the negative queries.)
    RERANKER_TYPE: str = "azure_semantic"
    MIN_RERANKER_SCORE: float = 0.0

    DATA_SOURCE_MOODLE_URL: str = "UNSET"
    DATA_SOURCE_MOODLE_TOKEN: str = "UNSET"

    DATA_SOURCE_MOOCHUP_HPI_URL: str = "UNSET"
    DATA_SOURCE_MOOCHUP_MOODLE_URL: str = "UNSET"

    VIMEO_PAT: str = "UNSET"

    # Chat runtime tuning — previously hardcoded constants in assistant.py / LLMs.py.
    MAX_CHAT_THREADS: int = 500  # BoundedMemorySaver: oldest thread evicted beyond this
    CHAT_HISTORY_LIMIT: int = 6  # messages kept per turn when rebuilding graph state
    GWDG_TIMEOUT_SECONDS: int = 7  # fallback to Azure if GWDG hasn't responded by then
    GWDG_UNAVAILABLE_RESET_SECONDS: int = 60 * 5  # window before retrying GWDG again

    @field_validator("ENVIRONMENT")
    def validate_ENVIRONMENT(cls, value: str) -> str:
        if value not in ["DEV", "PRODUCTION"]:
            raise ValueError("ENVIRONMENT must be DEV or PRODUCTION")
        return value

    @field_validator("RERANKER_TYPE")
    def validate_RERANKER_TYPE(cls, value: str) -> str:
        allowed = ("llm", "azure_semantic", "bge")
        if value not in allowed:
            raise ValueError(f"RERANKER_TYPE must be one of {allowed}, got {value!r}")
        return value

    @field_validator("REST_API_KEYS", mode="before")
    def transform_REST_API_KEYS(cls, value: list[str] | str) -> list[str]:
        if type(value) == str:
            # json elements must be double quoted, but are single quoted through terraform
            value = json.loads(value.replace("'", '"'))
        if type(value) != list:
            raise ValueError("REST_API_KEYS must be a list of strings.")
        return value

    @staticmethod
    def append_variable(kwargs: Any, variable_key: str, secret_client: SecretClient, class_variable: str = "") -> Any:
        """Appends a variable to the kwargs dictionary. If the variable is not set in the environment, it will be fetched from the key vault.
        Use class_variable to set a different class variable in the kwargs dictionary, than the variable_key name."""
        kwargs_key = class_variable if class_variable else variable_key

        if os.getenv(variable_key) is not None:
            kwargs[kwargs_key] = os.getenv(variable_key)
        else:
            try:
                # Azure Key Vault does not allow underscores in the key name but hyphens
                variable_key = variable_key.replace("_", "-")
                secret = secret_client.get_secret(variable_key)
                kwargs[kwargs_key] = secret.value
            except ResourceNotFoundError:
                logging.debug(f"Secret {variable_key} not found in the key vault, it will be unset.")
        # otherwise default value from pydantic model is used
        return kwargs

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Helper class for environment variables. Loads production variables if production=True, or env variable 'ENVIRONMENT' is set to 'PRODUCTION'"""
        if find_dotenv():
            load_dotenv(override=True)
        else:
            logging.warning("No .env file found.")

        # Using Azure Key Vault when secrets are not set through environment variables
        key_vault_name = os.environ.get("KEY_VAULT_NAME", "kicwa-keyvault-lab")
        key_vault_uri = f"https://{key_vault_name}.vault.azure.net/"
        credential = DefaultAzureCredential()
        secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)

        # Environment setup
        for key in self.model_json_schema()["properties"].keys():
            if key not in kwargs.keys():
                self.append_variable(kwargs, variable_key=key, secret_client=secret_client)
        # Use default ENVIRONMENT if not loaded from Key Vault or env
        environment = kwargs.get("ENVIRONMENT", "DEV")

        # Variables with different names
        if environment == "PRODUCTION":
            self.append_variable(
                kwargs,
                variable_key="DATA_SOURCE_PRODUCTION_MOODLE_URL",
                secret_client=secret_client,
                class_variable="DATA_SOURCE_MOODLE_URL",
            )
            self.append_variable(
                kwargs,
                variable_key="DATA_SOURCE_PRODUCTION_MOODLE_TOKEN",
                secret_client=secret_client,
                class_variable="DATA_SOURCE_MOODLE_TOKEN",
            )
        else:
            self.append_variable(
                kwargs,
                variable_key="DATA_SOURCE_PRODUCTION_MOODLE_URL",
                secret_client=secret_client,
                class_variable="DATA_SOURCE_MOODLE_URL",
            )
            self.append_variable(
                kwargs,
                variable_key="DATA_SOURCE_PRODUCTION_MOODLE_TOKEN",
                secret_client=secret_client,
                class_variable="DATA_SOURCE_MOODLE_TOKEN",
            )
        super().__init__(*args, **kwargs)

    @staticmethod
    def check_env():
        for attr, value in EnvHelper().__dict__.items():
            if value == "":
                logging.warning(f"{attr} is not set in the environment variables.")

    def __getattribute__(self, name: str):
        """Since not all environment variables are always required or set, this will raise an error if the attribute is not set during runtime."""
        value = object.__getattribute__(self, name)
        if value == "UNSET":
            raise AttributeError(f"{name} is requested but no value was provided.")
        return value

    def get_REST_API_KEYS(self):
        if len(self.REST_API_KEYS) == 0:
            raise AttributeError("REST_API_KEYS is requested but no value was provided.")
        return self.REST_API_KEYS


env = EnvHelper()
os.environ["LANGFUSE_PUBLIC_KEY"] = env.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = env.LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = env.LANGFUSE_HOST

if __name__ == "__main__":
    print(env.get_REST_API_KEYS())
