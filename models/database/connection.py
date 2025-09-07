"""
Database connection and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from config.settings import settings
from models.database.models import Base
import asyncio

# Synchronous engine for regular operations
sync_engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Asynchronous engine for async operations
if settings.DATABASE_URL.startswith("sqlite"):
    # For SQLite, use aiosqlite for async operations
    async_database_url = settings.DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")
    async_engine = create_async_engine(
        async_database_url,
        echo=settings.DATABASE_ECHO,
        poolclass=NullPool,
    )
else:
    # For PostgreSQL
    async_database_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    async_engine = create_async_engine(
        async_database_url,
        echo=settings.DATABASE_ECHO,
        poolclass=NullPool,
    )

# Session factories
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=sync_engine
)

AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


def get_db() -> Session:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncSession:
    """Dependency to get async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def create_tables():
    """Create all tables"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables():
    """Drop all tables"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


def create_tables_sync():
    """Create all tables synchronously"""
    Base.metadata.create_all(bind=sync_engine)


def drop_tables_sync():
    """Drop all tables synchronously"""
    Base.metadata.drop_all(bind=sync_engine)


if __name__ == "__main__":
    # Create tables when run directly
    create_tables_sync()
    print("Database tables created successfully!")
