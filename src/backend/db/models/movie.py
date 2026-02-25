from db.base import Base
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship


class Movie(Base):
    __tablename__ = 'movies'
    id = Column(Integer, autoincrement=True, primary_key=True, unique=True)
    title = Column(String)
    year = Column(Integer)
    other_data = Column(JSONB)
    graph_id = Column(Integer, ForeignKey('graph.id'))
    node = relationship('Graph', back_populates='movie')
    embeddings = relationship('Embedding', back_populates='movie')
