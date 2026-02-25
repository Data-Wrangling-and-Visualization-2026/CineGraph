from settings import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(settings.db.db_url, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, autocommit=False, autoflush=False)

def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()