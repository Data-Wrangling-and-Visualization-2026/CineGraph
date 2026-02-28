from multiprocessing import cpu_count

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from settings import settings


class MoviesGroupsNames(BaseModel):
    """Unique names for each group of movies.
    Each name logically describes the idea of grouping
    """
    names: list[str]


llm = ChatLlamaCpp(
    model_path=settings.name_creator.model_path,
    n_ctx=8192*2,
    n_gpu_layers=-1,
    n_batch=512,
    max_tokens=1024,
    n_threads=cpu_count() - 1,
    temperature=0.53,
    top_p=0.85,
    repeat_penalty=1.15,
    verbose=False,
)
structured_llm = llm.with_structured_output(MoviesGroupsNames)

system_prompt = """You are a semantic taxonomy expert.

Task:
Generate a unique, semantically accurate name for each provided group of movies. You DO NOT create groups, you MUST GIVE ONLY NAME
for already grouped movies.

Each group contains only movie titles.

Rules:
1. Each name must describe the shared theme, franchise, narrative focus, genre subtype, or defining concept.
2. Maximum 4 words per name.
3. Use Title Case.
4. Names must be unique across all groups.
5. Do NOT use vague labels like:
   - Drama Collection
   - Action Movies
   - Miscellaneous Films
   - Classic Movies
6. Avoid generic single-word genres unless strongly specific.
7. If unsure, infer the strongest shared concept.
8. If clearly a franchise, use the franchise name.
9. The number of names MUST equal the number of groups.
10. Do not explain anything.
11. Output only the structured response.

Ensure each name is concise, specific, and semantically grounded in the movies.
"""


def validate_names(names: list[str]):
    if names is None or len(names) == 0:
        return False

    for name in names:
        if len(name.split()) > 4:
            return False

    return True


def generate_context_aware_node_name(groups: list[list[str]]) -> str:
    message = 'Create names for the following groups:\nGroups:\n\n'
    for idx, group in enumerate(groups):
        message += f'Group {idx + 1}:\n'
        message += '\n'.join(group)
        message += '\n\n'

    message += f'Provide exactly {len(groups)} names in the JSON format specified.\nJSON Output:\n'
    messages = [SystemMessage(system_prompt), HumanMessage(message)]

    for _ in range(7):

        try:
            response = structured_llm.invoke(messages)
            names = response.names
            print(names)
            if len(set(names)) == len(groups) and validate_names(names):
                return [name.strip().replace(' ', '_') for name in names]

        except OutputParserException as e:
            print(e)

    raise ValueError('Failed to generate unique group names after retries')
