from pydantic_settings import BaseSettings
import os

class Config(BaseSettings):
    # Static settings
    projects_dir: str = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "projects"))
    faiss_index_dir: str = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_indexes"))
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    llm_temperature: float = 0.2
    retriever_k: int = 3
    
    # Add model cache directory
    model_cache_dir: str = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "cache"))
    
    # AWS settings to be loaded from .env
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_default_region: str

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Instantiate the settings
settings = Config()