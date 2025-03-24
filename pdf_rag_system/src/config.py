# src/config.py
import os

class Config:
    # File paths
    DATA_DIR = "data"
    STORAGE_DIR = "storage"
    VECTOR_STORE_DIR = os.path.join(STORAGE_DIR, "chroma_db")
    PROCESSED_PDFS_FILE = os.path.join(STORAGE_DIR, "processed_pdfs.json")
    
    # Text splitting parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Embedding model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Retrieval parameters
    DEFAULT_RETRIEVAL_K = 4
    
    # LLM parameters
    LLM_TEMPERATURE = 0

# Allow environment variables to override default config
def load_env_config():
    config = Config()
    
    # Override with environment variables if available
    if os.environ.get("DATA_DIR"):
        config.DATA_DIR = os.environ.get("DATA_DIR")
    
    if os.environ.get("STORAGE_DIR"):
        config.STORAGE_DIR = os.environ.get("STORAGE_DIR")
        config.VECTOR_STORE_DIR = os.path.join(config.STORAGE_DIR, "chroma_db")
        config.PROCESSED_PDFS_FILE = os.path.join(config.STORAGE_DIR, "processed_pdfs.json")
    
    if os.environ.get("CHUNK_SIZE"):
        config.CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
    
    if os.environ.get("CHUNK_OVERLAP"):
        config.CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP"))
    
    if os.environ.get("EMBEDDING_MODEL"):
        config.EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
    
    if os.environ.get("DEFAULT_RETRIEVAL_K"):
        config.DEFAULT_RETRIEVAL_K = int(os.environ.get("DEFAULT_RETRIEVAL_K"))
    
    return config

# Create a singleton instance
config = load_env_config()