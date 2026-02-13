import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class ScraperSettings(BaseModel):
    url: str = 'https://subslikescript.com'

    output_folder: str = os.path.join(
        BASE_DIR, 'data' # -> /src/data/
    )

    # Do not use stateful mode inside Docker
    headless: bool = True

    delay: int = 3

    num_workers: int = 4


class Settings(BaseModel):
    scraper: ScraperSettings = ScraperSettings()
    base_dir: str = BASE_DIR


settings = Settings()