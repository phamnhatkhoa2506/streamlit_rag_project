from langchain_huggingface import HuggingFaceEmbeddings
from config import envConfig


redis_embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
