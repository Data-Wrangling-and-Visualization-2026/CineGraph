import os
from contextlib import asynccontextmanager

from api.admin_panel import router
from api.base_models import MovieResponse, MovieSubmission, NodeWithChildren
from db.repositories.graph_repo import GraphRepository
from db.session import get_db
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from services.movie_service import place_into_verification_queue, validate_movie
from settings import settings
from sqlalchemy.ext.asyncio import AsyncSession


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Run only once, on app startup. Ensures ./movies_for_validation exists

    Args:
        app (FastAPI): app
    """
    os.makedirs(
        settings.movie_validator.path_to_files,
        exist_ok=True
    )
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)
app.add_middleware(
    CORSMiddleware,
    # allow_origins=[f"http://localhost:{os.environ['FRONT_PORT']}"],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/add_movie")
async def add_movie(movie_in: MovieSubmission):
    """
    Accepts movie data and subtitles, validates them, and saves to a directory
    for future user validation.
    """
    try:
        validate_movie(movie_in)

        movie_data = movie_in.model_dump()

        await place_into_verification_queue(movie_data)

        return {
            "status": "success",
            "message": "Movie saved for user validation.",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
