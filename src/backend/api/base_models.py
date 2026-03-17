from pydantic import BaseModel
from typing import List, Optional, Any


# Movie schemas
class MovieBase(BaseModel):
    title: str
    year: Optional[int] = None
    other_data: Optional[dict] = None


class MovieCreate(MovieBase):
    pass

class MovieResponse(MovieBase):
    id: int
    graph_id: int

    class Config:
        from_attributes = True


# Node (Graph) schemas
class NodeBase(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None


class NodeCreate(NodeBase):
    pass

class NodeResponse(NodeBase):
    id: int
    path: str
    children_count: int

    class Config:
        from_attributes = True


class NodeWithChildren(NodeResponse):
    # Changed from List[NodeResponse] to List[int]
    children_nodes: List[int] = []  # Now accepts list of node IDs
    # Changed from List[MovieResponse] to List[int]
    movies: List[int] = []  # Now accepts list of movie IDs