import json
from pathlib import Path
from urllib.request import Request

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from services.movie_service import process_movie
from settings import settings

security = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != settings.admin.login \
        or credentials.password != settings.admin.password:
        raise HTTPException(status_code=401)


MOVIES_DIR = Path(settings.movie_validator.path_to_files)
router = APIRouter(prefix='/admin')
templates = Jinja2Templates(directory='./api/templates')

@router.get('/movies')
async def list_movies():
    files = list(MOVIES_DIR.glob("*.json"))
    return [f.name for f in files]


@router.get('/movies/{filename}')
async def get_movie_file(filename: str):
    file_path = MOVIES_DIR / filename

    if not file_path.exists():
        raise HTTPException(404, 'File not found')

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@router.post('/movies/{filename}/approve')
async def approve_movie(filename: str):
    file_path = MOVIES_DIR / filename

    if not file_path.exists():
        raise HTTPException(404, 'File not found')

    with open(file_path, "r", encoding="utf-8") as f:
        movie_data = json.load(f)

    await process_movie(movie_data)

    file_path.unlink()

    return {'status': 'approved'}


@router.delete('/movies/{filename}')
async def reject_movie(filename: str):
    file_path = MOVIES_DIR / filename

    if not file_path.exists():
        raise HTTPException(404, 'File not found')

    file_path.unlink()

    return {'status': 'deleted'}


@router.get('/', dependencies=[Depends(verify_admin)])
async def admin_panel(request: Request):
    return templates.TemplateResponse(
        'admin.html',
        {'request': request}
    )