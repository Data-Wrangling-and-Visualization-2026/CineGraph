import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv() # for manual use (please, ensure that there is .env file in the same folder)

BASE_DIR = Path(__file__).resolve().parent


class ScraperSettings(BaseModel):
    url: str = Field(
        default='https://subslikescript.com',
        description="Base URL for scraping"
    )

    output_folder: str = Field(
        default=os.path.join(BASE_DIR, 'data'),
        description="Directory where scraped files will be saved"
    )

    # Do not set False if using Docker
    headless: bool = Field(
        default=False,
        description="Run browser in headless mode (disable UI)"
    )

    delay: int = Field(
        default=3,
        description="Delay between requests in seconds"
    )

    num_workers: int = Field(
        default=4,
        description="Number of concurrent workers"
    )

    offset: int = Field(
        default=386,
        description="End page for the first worker at the last run"
    )


class EmotionAnalyzerSettings(BaseModel):
    input_path: str = Field(
        default='./data',
        description="Path to preprocessed data"
    )

    output_path: str = Field(
        default='./emotion_analysis/embeddings',
        description="Path to files with embeddings"
    )


class PreprocessorSettings(BaseModel):
    use_hugging_face: bool = Field(
        default=False,
        description="Flag to use propitiate HuggingFace model"
    )

    local_model_path: str = Field(
        default='./emotion_analysis/weights/mistral_prep/model.gguf',
        description="Path to file with local model"
    )

    model: str = Field(
        default='Qwen/Qwen2.5-7B-Instruct',
        description="Model name"
    )

    input_path: str = Field(
        default='./data',
        description="Path to raw subtitles"
    )

    output_path: str = Field(
        default=Path('./preprocessing/ready_data'),
        description="Path to folder with preprocessed data"
    )

    chunk_size: int = Field(
        default=4096,
        description="Each text is divided into chunks since " \
        "entire sequence can not fit into context"
    )

    offset: int = Field(
        default=0,
        description="Determines how much first files will be skipped "
        "(in case some have already been preprocessed)"
    )


class DBSettings(BaseModel):
    db_url: str = Field(
        default=os.environ['DB_URL'],
        description="DB connection string"
    )


class GraphSettings(BaseModel):
    max_depth: int = Field(
        default=5,
        description="Maximum graph depth (excluding the highest node (root))"
    )

    min_samples_leaf: int = Field(
        default=6,
        description="Minimum number of movies in each leaf"
    )

    max_nodes: int = Field(
        default=800,
        description="Maximum number of nodes in the graph (important for KMeans)"
    )

    target_leaf_size: int = Field(
        default=50,
        description="Desired number of movies in each leaf"
    )

    # Important - with current algorithm min/max_fanout can be violated for very small
    # number of nodes. To understand the reason check the algorithm description
    # in ./clustering/graph_creator.py
    min_fanout: int = Field(
        default=3,
        description="Minimal number of successors for each node"
    )

    max_fanout: int = Field(
        default=8,
        description="Maximum number of successors for each node"
    )


class NameCreatorSettings(BaseModel):
    model_path: str = Field(
        default='./emotion_analysis/weights/qwen/model.gguf',
        description="Path to model"
    )


class APISettings(BaseModel):
    app_path: str = Field(
        default='api.api:app',
        description='Path to file with FastAPI app'
    )

    host: str = Field(
        default='0.0.0.0',
    )

    port: int = Field(
        default=int(os.environ['API_PORT']),
    )


class Settings(BaseModel):
    scraper: ScraperSettings = ScraperSettings()
    emotion_analyzer: EmotionAnalyzerSettings = EmotionAnalyzerSettings()
    preprocessor: PreprocessorSettings = PreprocessorSettings()
    name_creator: NameCreatorSettings = NameCreatorSettings()
    db: DBSettings = DBSettings()
    graph: GraphSettings = GraphSettings()
    api: APISettings = APISettings()
    base_dir: str = BASE_DIR


settings = Settings()