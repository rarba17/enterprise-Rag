import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from dotenv import load_dotenv

load_dotenv()

# We use the 'postgresql+asyncpg' driver for FastAPI performance
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:password@localhost:5432/rag_db")

# 1. Create the Async Engine
# 'echo=True' is helpful for debugging during your project development
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600
    )

# 2. Create a Session Factory
# expire_on_commit=False prevents issues when accessing objects after a commit
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# 3. Base class for our models
class Base(DeclarativeBase):
    pass

# 4. Dependency to get DB session in FastAPI routes
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()