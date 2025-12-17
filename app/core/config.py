from pydantic_settings import BaseSettings, SettingsConfigDict
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    openai_api_key: str
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    data_path: str = "data"
    vectorstore_path: str = "data/vectorstore"
    demo_dataset_path: str = "data/demo_dataset.txt"
    chunk_size_default: int = 512
    chunk_overlap_default: int = 50
    debug: bool = False
    project_name: str = "RAG-ML"

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )


settings = Settings()
