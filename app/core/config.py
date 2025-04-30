import os
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Warehouse Task Optimization Platform"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "supersecretkey")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Database
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "warehouse_platform")
    DATABASE_URI: Optional[str] = None

    # Redis/Celery
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: str = os.getenv("REDIS_PORT", "6379")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000", "http://localhost", "http://127.0.0.1:5175",  "https://husslify.com", "https://www.husslify.com", "https://capstone-frontend.pages.dev", "https://f0f288bb.capstone-frontend.pages.dev", "https://api.husslify.com"]

    # Email
    EMAIL_SENDER: str = os.getenv("EMAIL_SENDER", "")
    SENDGRID_API_KEY: str = os.getenv("SENDGRID_API_KEY", "")

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.DATABASE_URI = f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


settings = Settings() 
