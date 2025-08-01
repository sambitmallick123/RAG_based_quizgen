import os
from pathlib import Path

class Config:
    """Configuration settings for the RAG Quiz Generator"""
    
    # Application settings
    APP_HOST = "0.0.0.0"
    APP_PORT = 13000
    
    # Model settings
    MODEL_NAME = "./models/Llama-3.2-3B-Instruct"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # ChromaDB settings
    CHROMA_DB_DIR = "./chroma_db"
    COLLECTION_NAME = "quiz_documents"
    
    # Text processing settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    MAX_CONTEXT_LENGTH = 1000
    
    # Quiz generation settings
    MAX_QUESTIONS = 20
    MIN_QUESTIONS = 1
    DEFAULT_QUESTIONS = 5
    
    # Model generation settings
    MAX_NEW_TOKENS = 300
    TEMPERATURE = 0.7
    TOP_P = 0.9
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist"""
        Path(cls.CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)

# Initialize directories on import
Config.ensure_directories()

