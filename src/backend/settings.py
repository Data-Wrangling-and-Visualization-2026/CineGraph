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
        default=2,
        description="Number of concurrent workers"
    )

    offset: int = Field(
        default=77,
        description="End page for the first worker at the last run"
    )


class EmotionAnalyzerSettings(BaseModel):
    input_path: str = './data'
    output_path: str = './emotion_analysis/embeddings'


class PreprocessorSettings(BaseModel):
    use_hugging_face: bool = False
    local_model_path: str = './emotion_analysis/weights/mistral_prep/model.gguf'
    model: str = 'Qwen/Qwen2.5-7B-Instruct'
    input_path: str = './data'
    output_path: str = './preprocessing/ready_data'
    chunk_size: int = 4096
    offset: int = 0


class Settings(BaseModel):
    scraper: ScraperSettings = ScraperSettings()
    emotion_analyzer: EmotionAnalyzerSettings = EmotionAnalyzerSettings()
    preprocessor: PreprocessorSettings = PreprocessorSettings()
    base_dir: str = BASE_DIR


settings = Settings()