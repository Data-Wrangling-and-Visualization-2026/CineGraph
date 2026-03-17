from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
import os

from db.repositories.graph_repo import GraphRepository
from db.session import get_db
from api.base_models import NodeWithChildren, NodeResponse, MovieResponse

app = FastAPI()


@app.get("/")
async def get_logo():
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")

    if os.path.exists(logo_path):
        return FileResponse(
            logo_path,
            media_type="image/png",
            filename="logo.png"
        )
    return {"error": "Logo not found"}


@app.get("/graph", response_model=NodeWithChildren)
async def get_graph_node(
        node: int = Query(..., description="Node ID"),
        session: AsyncSession = Depends(get_db),
):
    repo = GraphRepository(session)

    try:
        result = await repo.get_immediate_children(node)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Get the node object
    node_obj = result["node"]


    return NodeWithChildren(
        id=node_obj.id,
        name=node_obj.name,
        type=node_obj.type,
        path=node_obj.path,
        children_count=node_obj.children_count,
        children_nodes=result["children_nodes"],
        movies=result["movies"],
    )


@app.get("/movie", response_model=MovieResponse)
async def get_movie(
        id: int = Query(..., description="Movie ID"),
        session: AsyncSession = Depends(get_db),
):
    repo = GraphRepository(session)

    movie = await repo.get_movie(id)

    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")

    return MovieResponse.model_validate(movie)