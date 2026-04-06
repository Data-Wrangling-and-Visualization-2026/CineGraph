from db.base import Base
from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Column, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship


class Movie(Base):
    __tablename__ = 'movies'
    id = Column(Integer, autoincrement=True, primary_key=True, unique=True)
    title = Column(String(100))
    year = Column(Integer)
    other_data = Column(JSONB)
    graph_id = Column(Integer, ForeignKey('graph.id'))
    emotion_arc = Column(VECTOR)
    node = relationship('Graph', back_populates='movie')
    embeddings = relationship('Embedding', back_populates='movie')

    __table_args__ = (
        Index('movies_embedding_hnsw_idx', emotion_arc, postgresql_using='hnsw',
              postgresql_ops={'emotion_arc': 'vector_cosine_ops'}),
    )