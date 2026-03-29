import asyncio
import json
import os
import uuid
from typing import Any

import aiofiles
import httpx
import numpy as np
import pandas as pd
from db.repositories.graph_repo import GraphRepository
from db.session import get_db
from settings import settings

HF_URL = os.environ['HF_API_URL']
emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def construct_embedding(embeddings: pd.DataFrame) -> np.ndarray[float]:
    acts = np.array_split(embeddings[emotions].values, 3)

    data = [
        acts[i].mean(axis=0) for i in range(3)
    ]

    data.append(
        embeddings[emotions].std(axis=0).values
    )

    return np.concat(data)


async def attach_to_best_parent_node(movie_metadata: dict[Any, str], embeddings: pd.DataFrame) -> None:
    centroid = construct_embedding(embeddings)

    async for db in get_db():
        repo = GraphRepository(db)

        nodes = await repo.get_all_nodes()

        best_distance = float('inf')
        best_parent = None
        for node in nodes:
            distance = np.linalg.norm(node.centroid - centroid)

            if distance < best_distance:
                best_distance = distance
                best_parent = node

        if best_parent is None:
            raise RuntimeError('Could not find suitable parent!')

        movie = await repo.add_movie(
            graph_id=best_parent.id,
            title=movie_metadata['title'],
            year=movie_metadata['year'],
            vectors=embeddings[emotions].values
        )

        print(movie.id)


async def place_into_verification_queue(movie: dict[str, Any]) -> None:
    dir_path = settings.movie_validator.path_to_files

    await asyncio.to_thread(os.makedirs, dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, f"{uuid.uuid4()}.json")

    data = await asyncio.to_thread(
        json.dumps, movie, indent=4, ensure_ascii=False
    )

    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(data)


async def process_movie(movie: dict[str, Any]) -> None:
    async with httpx.AsyncClient(timeout=120.0) as client:
        clean_resp = await client.post(
            f'{HF_URL}/clean',
            json={'text': movie['subtitles']}
        )

        if clean_resp.status_code != 200:
            raise RuntimeError(f'Clean API error: {clean_resp.text}')

        cleaned = clean_resp.json()['text']

        analyze_resp = await client.post(
            f'{HF_URL}/analyze',
            json={'text': cleaned}
        )

        if analyze_resp.status_code != 200:
            raise RuntimeError(f'Analyze API error: {analyze_resp.text}')

    result = analyze_resp.json()

    emotions_df = pd.DataFrame(result["data"])
    print(emotions_df)

    await attach_to_best_parent_node(
        movie_metadata=movie,
        embeddings=emotions_df
    )