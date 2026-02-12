import os
from pathlib import Path

from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent


class ScraperSettings(BaseModel):
    url: str = 'https://subslikescript.com'

    output_folder: str = os.path.join(
        BASE_DIR, 'data' # -> /src/data/
    )

    headless: bool = True

    delay: int = 6

    num_workers: int = 2


class Settings(BaseModel):
    scraper: ScraperSettings = ScraperSettings()
    base_dir: str = BASE_DIR


settings = Settings()