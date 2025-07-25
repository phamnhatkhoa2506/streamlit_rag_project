from pydantic_settings import BaseSettings


class EnvConfig(BaseSettings):
    GOOGLE_API_KEY: str | None = None
    LANGCHAIN_API_KEY: str | None = None
    TAVILY_API_KEY: str | None = None
    PINECONE_API_KEY: str | None = None
    HUGGINGFACE_TOKEN: str | None = None

    REDIS_URL: str | None = None
    REDIS_HOST: str | None = None
    REDIS_PORT: int | None = None

    class Config:
        env_file = '.env'


envConfig = EnvConfig()