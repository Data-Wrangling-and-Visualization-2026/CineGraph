from multiprocessing import cpu_count

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field, create_model
from settings import settings

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

system_prompt = """You are an expert in narrative psychology, film taxonomy, and story arcs.

Task:
You will be given several distinct groups of movies. You MUST generate exactly one unique, evocative sub-category name for EVERY group.

Rules:
1. Focus on the Emotional Shift and the specific Movie Titles to find the narrative vibe.
2. If multiple groups have similar emotional shifts, use the Movie Titles to find the thematic difference.
3. Maximum 4 words per name. Use Title Case (e.g., "Tragic Downfalls", "High-Octane Climaxes").
4. DO NOT use generic terms like "Collection", "Movies", or repeat the parent's exact name.
5. Every single name you generate must be distinct from the others.
6. Do NOT summarize the groups into one name. You must name each group individually.
"""

def validate_names(names: list[str], expected_length: int):
    if names is None or len(names) != expected_length:
        return False

    for name in names:
        if len(name.split()) > 4:
            return False

    return True

def clean_titles(titles: list[str]) -> list[str]:
    return[
        title.encode('ascii', errors='ignore').decode('ascii')
        for title in titles
    ]


def generate_context_aware_node_name(parent_name: str, groups: list[dict]) -> list[str]:
    num_groups = len(groups)

    # ------------------------------------------------------------------
    # 1. DYNAMICALLY CREATE THE SCHEMA
    # This creates a schema like:
    # { "group_1": "...", "group_2": "...", ..., "group_N": "..." }
    # This physically forces the Llama.cpp grammar to output N items.
    # ------------------------------------------------------------------
    fields = {
        f"group_{i+1}": (str, Field(description=f"The unique 2-4 word category name specifically for Group {i+1}"))
        for i in range(num_groups)
    }
    DynamicNamesModel = create_model('DynamicNamesModel', **fields)

    # Bind the dynamic model for this specific call
    dynamic_structured_llm = llm.with_structured_output(DynamicNamesModel)

    # 2. Build the prompt
    message = f"Parent Category: '{parent_name}'\n\n"
    message += "Create specific sub-category names for the following distinct groups:\n\n"

    for idx, group in enumerate(groups):
        message += f'Group {idx + 1}:\n'
        message += f"- Emotional Shift from Parent: {group['shift']}\n"
        message += f"- Representative Movies: {', '.join(clean_titles(group['titles']))}\n\n"

    messages =[SystemMessage(content=system_prompt), HumanMessage(content=message)]

    # 3. Execute with retries
    for attempt in range(5):
        try:
            response = dynamic_structured_llm.invoke(messages)

            # response is a dynamic Pydantic object. .model_dump() turns it into a dict
            # .values() gets just the generated names in order
            names = list(response.model_dump().values())

            if len(set(names)) == num_groups and validate_names(names, num_groups):
                return[name.strip().replace(' ', '_') for name in names]
            else:
                print(f"Attempt {attempt + 1} failed. Validation or uniqueness failed. Generated: {names}")

        except Exception as e:
            print(f'Retry LLM attempt {attempt+1} failed: {e}')

    print(f"Failed to generate valid names for {parent_name}. Using deterministic fallbacks.")
    return[f"{parent_name.replace(' ', '_')}_Subgroup_{i+1}" for i in range(num_groups)]