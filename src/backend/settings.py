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
        default=True,
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


class Settings(BaseModel):
    scraper: ScraperSettings = ScraperSettings()
    base_dir: str = BASE_DIR


settings = Settings()