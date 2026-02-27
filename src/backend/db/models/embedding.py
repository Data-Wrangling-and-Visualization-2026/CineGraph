from db.base import Base
from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Column, ForeignKey, Index, Integer
from sqlalchemy.orm import relationship


class Embedding(Base):
    __tablename__ = 'embeddings'
    id = Column(Integer, autoincrement=True, primary_key=True, unique=True)
    window_id = Column(Integer)
    movie_id = Column(Integer, ForeignKey('movies.id'))
    embedding = Column(VECTOR)
    movie = relationship('Movie', back_populates='embeddings')

    __table_args__ = (
        Index('ix_embeddings_vector', embedding, postgresql_using='hnsw',
              postgresql_ops={'embedding': 'vector_cosine_ops'}),
    )