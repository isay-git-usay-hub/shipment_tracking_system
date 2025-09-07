"""
Configuration management for Maersk Shipment AI System
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    API_HOST: str = Field(default="localhost", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_DEBUG: bool = Field(default=True, env="API_DEBUG")
    API_RELOAD: bool = Field(default=True, env="API_RELOAD")

    # Database Configuration
    DATABASE_URL: str = Field(env="DATABASE_URL")
    DATABASE_ECHO: bool = Field(default=False, env="DATABASE_ECHO")

    # Redis Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # Local LLM Configuration
    LOCAL_LLM_MODEL_PATH: str = Field(default="models/llm", env="LOCAL_LLM_MODEL_PATH")
    LOCAL_LLM_MAX_TOKENS: int = Field(default=2000, env="LOCAL_LLM_MAX_TOKENS")
    LOCAL_LLM_TEMPERATURE: float = Field(default=0.7, env="LOCAL_LLM_TEMPERATURE")

    # SendGrid Configuration
    SENDGRID_API_KEY: str = Field(env="SENDGRID_API_KEY")
    FROM_EMAIL: str = Field(default="noreply@maersk.com", env="FROM_EMAIL")

    # JWT Configuration
    SECRET_KEY: str = Field(env="SECRET_KEY")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="logs/app.log", env="LOG_FILE")

    # ML Model Configuration
    MODEL_RETRAIN_INTERVAL_HOURS: int = Field(default=24, env="MODEL_RETRAIN_INTERVAL_HOURS")
    MODEL_PREDICTION_BATCH_SIZE: int = Field(default=100, env="MODEL_PREDICTION_BATCH_SIZE")
    MODEL_CACHE_TTL_MINUTES: int = Field(default=60, env="MODEL_CACHE_TTL_MINUTES")

    # External APIs
    WEATHER_API_KEY: Optional[str] = Field(default=None, env="WEATHER_API_KEY")
    PORT_API_KEY: Optional[str] = Field(default=None, env="PORT_API_KEY")

    # Dashboard Configuration
    DASHBOARD_AUTO_REFRESH_SECONDS: int = Field(default=30, env="DASHBOARD_AUTO_REFRESH_SECONDS")
    DASHBOARD_MAX_RECORDS: int = Field(default=1000, env="DASHBOARD_MAX_RECORDS")

    # Celery Configuration
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")

    # CORS Configuration
    ALLOWED_HOSTS: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    ALLOWED_ORIGINS: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")

    # Model Paths
    MODEL_PATH: Path = Field(default=Path("models/trained"), env="MODEL_PATH")
    DATA_PATH: Path = Field(default=Path("data"), env="DATA_PATH")

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Create necessary directories
os.makedirs("logs", exist_ok=True)
os.makedirs(settings.MODEL_PATH, exist_ok=True)
os.makedirs(settings.DATA_PATH / "raw", exist_ok=True)
os.makedirs(settings.DATA_PATH / "processed", exist_ok=True)
os.makedirs(settings.DATA_PATH / "features", exist_ok=True)
