import random
from multiprocessing import cpu_count

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from settings import settings


class CategoryName(BaseModel):
    name: str = Field(description='A unique 2-4 word descriptive category name. Use Title Case.')

# Base model
llm = ChatLlamaCpp(
    model_path=settings.name_creator.model_path,
    n_ctx=8192,
    n_gpu_layers=-1,
    n_batch=512,
    max_tokens=1500,
    n_threads=cpu_count() - 1,
    temperature=0.3,
    top_p=0.9,
    repeat_penalty=1.05,
    verbose=False,
)
structured_llm = llm.with_structured_output(CategoryName)

system_prompt = """You are an expert in narrative psychology and film taxonomy.
Your task is to provide exactly ONE evocative 2-4 word subgenre or theme name.
Rules:
1. Maximum 4 words. Use Title Case (e.g., "Tragic Downfalls", "Heroic Journeys").
2. DO NOT use generic words like "Collection", "Movies", "Group", or "Cluster".
3. Capture the overarching narrative, mood, or structural vibe.
"""


def clean_titles(titles: list[str]) -> list[str]:
    """
    Cleans the titles from non-ascii symbols

    Args:
        titles (list[str]): titles

    Returns:
        list[str]: cleaned text
    """
    return[
        title.encode('ascii', errors='ignore').decode('ascii')
        for title in titles
    ]


def generate_context_aware_node_name(movies: list[str], leaf: bool = True) -> list[str]:

    message = ''
    if leaf:
        message = f"Look at the following movie titles. What specific narrative archetype or atmospheric theme \
            unites them?\n\nMovies: {', '.join(clean_titles(movies))}\n\nReturn ONLY the 2-4 word category name."
    else:
        message = f"Look at the following sub-categories. What broader overarching narrative theme encompasses \
            all of them?\n\nSub-categories: {', '.join(movies)}\n\nReturn ONLY the 2-4 word category name."

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=message)]

    for attempt in range(3):
        try:
            response = structured_llm.invoke(messages)
            name = response.name.strip()
            if 0 < len(name.split()) <= 6:
                return name.replace(" ", "_")
        except Exception as e:
            print(f'LLM Retry leaf {attempt+1}: {e}')

    return "Thematic_Cluster_" + str(random.randint(100, 999))
