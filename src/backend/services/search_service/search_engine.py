import asyncio
import os

import httpx
import numpy as np
from db.repositories.graph_repo import GraphRepository
from db.session import get_db
from google import genai
from settings import HF_URL

client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

template = """
## System
The user will describe the emotional journey they want from a movie.
Generate 5-8 short passages of dialogue/monologue that would appear
in a movie matching that description. Order them chronologically
to represent the emotional arc.

Output ONLY the passages, separated by ---

## User input
{}
"""

async def find_matching_movies(embedding: np.ndarray) -> list:
    async for db in get_db():
        repo = GraphRepository(db)

        movies = await repo.find_closest(embedding)

    return movies


async def find_best_match(user_request: str) -> list:
    prompt = template.format(user_request)

    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.5-flash",
        contents=prompt
    )
    generated_text = response.text.split('---')

    embeddings = []
    async with httpx.AsyncClient(timeout=10.0) as http_client:
        for part in generated_text:
            embedding = []

            analyze_resp = await http_client.post(
                f'{HF_URL}/analyze',
                json={'text': part}
            )

            if analyze_resp.status_code != 200:
                print(f'Analyze API error: {analyze_resp.text}')

            emotions = analyze_resp.json()['data'][0]
            for emotion in emotions.keys():
                if not emotion.startswith('window'):
                    embedding.append(emotions[emotion])

            embeddings.append(embedding)

    embeddings = np.array(embeddings, dtype=np.float32)
    emotion_arc = np.concat(
        [
            act.mean(axis=0)
            for act in np.array_split(
                embeddings,
                indices_or_sections=3,
                axis=0
            )
        ] + [embeddings.std(axis=0)],
    )

    return await find_matching_movies(emotion_arc)


    # {'data': [{'sadness': 0.9812568426132202, 'joy': 0.00961368065327406, 'love': 0.002246781252324581, 'anger': 0.004471870604902506, 'fear': 0.0021979548037052155, 'surprise': 0.00021283802925609052, 'window_id': 0, 'window_start': 0, 'window_end': 50}]}
# [[0.000941246107686311, 0.9895933866500854, 0.008238101378083229, 0.000711622997187078, 0.00031703486456535757, 0.00019864790374413133], [0.0004345218767412007, 0.9978641867637634, 0.00042106746695935726, 0.0005439280648715794, 0.0005025786231271923, 0.00023361002968158573], [0.0005728397518396378, 0.9962355494499207, 0.0009269893635064363, 0.0009183744550682604, 0.0011340421624481678, 0.00021223348448984325], [0.00035253551322966814, 0.031019331887364388, 0.9634017944335938, 0.0006860996945761144, 0.0005418940563686192, 0.003998341038823128], [0.8993925452232361, 0.09049384295940399, 0.0018087198259308934, 0.001214031595736742, 0.006680129561573267, 0.0004108271677978337], [0.00028564195963554084, 0.9982714653015137, 0.0008666364592500031, 0.000153884626342915, 0.000210080950637348, 0.00021238204499240965]]
# ["Another perfect sunrise over the bay. Some days, that's all you really need, isn't it? Just this quiet little moment.\n\n", "\n\nI used to think my days were just *fine*. And they are. But lately, I’ve been wondering… what if *fine* isn't the only option?\n\n", "\n\nYou know, when they said 'opportunity knocks,' I always pictured a big, dramatic bang. Never really thought it would sound like... a gently ringing doorbell.\n\n", "\n\nIt’s funny how a little push can send you soaring. I didn't realize how much I needed to just... lean in. And honestly? I'm loving the view from up here.\n\n", "\n\nI keep trying to find the words, but they all fall short. It’s not just a good feeling; it’s like coming home to a place you didn't even know you were missing.\n\n", "\n\nAnd here we are. This. All of this. It’s more than I ever imagined. It’s not just a happy ending, is it? It’s a beautiful, wide-open beginning. And I wouldn't trade a single moment."]