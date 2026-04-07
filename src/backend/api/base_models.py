from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# Movie schemas
class MovieBase(BaseModel):
    title: str
    year: Optional[int] = None
    other_data: Optional[dict] = None


class MovieCreate(MovieBase):
    pass

class EmbeddingResponse(BaseModel):
    window_id: int
    embedding: List[float]

    class Config:
        from_attributes = True


class MovieSubmission(MovieCreate):
    subtitles: str

class MovieResponse(MovieBase):
    id: int
    graph_id: int
    title: str
    embeddings: List[EmbeddingResponse]

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
    # path: str
    children_count: int

    class Config:
        from_attributes = True


class NodeWithChildren(NodeResponse):
    children_nodes: List[NodeResponse] = []
    movies: List[MovieResponse] = []  # Now accepts list of movie IDs


class MoviesResponse(BaseModel):
    movies: List[MovieResponse] = []


class SearchRequest(BaseModel):
    description: Union[str, List[float]] = Field(..., description='Either movie emotion description or vector [24]')

    @field_validator('description')
    def validate_vector(cls, v):
        if isinstance(v, list):
            if not all(isinstance(x, (float, int)) for x in v):
                raise ValueError("Vector must contain only numbers")
        return v
