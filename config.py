import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    DATA_PATH = "data/"  # Directory containing medical PDFs
    CHUNK_SIZE = 2000  # Optimized: Increased for fewer chunks
    CHUNK_OVERLAP = 100  # Optimized: Reduced overlap
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "llama3-70b-8192"
    FAISS_INDEX_PATH = "faiss_index"  # For saving/loading vector store