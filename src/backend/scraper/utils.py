import os

from settings import settings


def create_data_folder() -> None:
    os.makedirs(
        settings.scraper.output_folder,
        exist_ok=True
    )