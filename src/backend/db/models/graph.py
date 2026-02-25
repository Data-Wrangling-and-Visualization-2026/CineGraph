from db.base import Base
from sqlalchemy import Column, Index, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy_utils import LtreeType


class Graph(Base):
    __tablename__ = 'graph'
    id = Column(Integer, primary_key=True, autoincrement=True)
    path = Column(LtreeType, nullable=False)
    name = Column(String)
    type = Column(String)
    children_count = Column(Integer, default=0)
    movie = relationship('Movie', back_populates='node')

    __table_args__ = (
        Index('ix_path', path, postgresql_using='gist'),
    )
