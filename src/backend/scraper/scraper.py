import os
import re
from multiprocessing import Pool

from scraper.utils import create_data_folder
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from settings import settings


class Scraper:
    def __init__(self) -> None:

        create_data_folder()

        self.BASE_URL = settings.scraper.url
        self.output_folder = settings.scraper.output_folder
        self.delay = settings.scraper.delay
        self.num_workers = settings.scraper.num_workers


    def _setup_driver(self) -> WebDriver:
        chrome_opts = Options()

        if settings.scraper.headless:
            chrome_opts.add_argument('--headless')
        chrome_opts.add_argument('--disable-gpu')
        chrome_opts.add_argument('--no-sandbox')
        chrome_opts.add_argument("--disable-dev-shm-usage")
        chrome_opts.add_argument(
            'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' \
            '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )

        service = Service(executable_path='/usr/bin/chromedriver')
        driver = webdriver.Chrome(service=service, options=chrome_opts)

        return driver


    def _separate_year_and_title(self, text: str) -> tuple[str, str]:
        match = re.search(r'(.*)\s\((\d{4})\)$', text)
        if match:
            return match.group(1).strip(), match.group(2)
        return text.strip(), '1800'


    def _get_movie_links(self, driver: WebDriver, page: int) -> list[dict[str, str]]:
        print(f'Collecting links from page #{page}')

        movie_metadata = []
        url = f'{self.BASE_URL}/movies?page={page}'
        driver.get(url)

        try:
            WebDriverWait(driver, self.delay).until(
                EC.presence_of_element_located(
                    (By.CLASS_NAME, 'scripts-list')
                    )
            )

            links = driver.find_elements(By.CSS_SELECTOR, '.scripts-list li a')

            for link in links:
                text = link.text
                href = link.get_attribute('href')

                title, year = self._separate_year_and_title(text)

                movie_metadata.append({
                    'title': title,
                    'year': year,
                    'url': href
                })
        except Exception as e:
            print(f'Error reading page {url}: {e}')

        return movie_metadata


    def _save(self, movie: dict[str, str], text: str) -> None:
        path = os.path.join(
            self.output_folder,
            f'{movie["title"]}_{movie["year"]}.txt'.replace(' ', '_'),
        )

        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f'{movie["title"]} {movie["year"]} saved!')


    def _scrape_subtitles(self, driver: WebDriver, movie: dict) -> str:
        print(f'Scraping {movie["title"]} {movie["year"]}')

        driver.get(movie['url'])

        full_script = ''

        try:
            WebDriverWait(driver, self.delay).until(
                EC.presence_of_element_located((By.ID, 'cue-app'))
            )

            lines = driver.find_elements(By.CLASS_NAME, 'cue-line')

            if not lines:
                print(f'No subtitles for {movie["title"]} {movie["year"]}')

            lines = [line.text for line in lines]
            full_script = '\n'.join(lines)

            self._save(movie, full_script)
        except Exception as e:
            print(f'Error scraping detail page for {movie["title"]}: {e}')

        return full_script


    def _start_worker(self, page_range: list[int], worker_id: int = 0) -> None:
        driver = self._setup_driver()

        try:
            all_movies = []

            for page in page_range:
                movies_on_page = self._get_movie_links(driver, page)
                for movie in movies_on_page:
                    self._scrape_subtitles(driver, movie)

                all_movies.extend(movies_on_page)

        finally:
            driver.quit()
            print(f'Worker №{worker_id} finished scraping!')


    def start_scraping(self) -> None:
        self._setup_driver().quit()

        start_page = 1
        end_page = 2029

        all_pages = list(range(start_page, end_page + 1))
        chunk_size = (len(all_pages) + self.num_workers - 1) // self.num_workers
        page_mapping = [all_pages[i : i + chunk_size] for i in range(0, len(all_pages), chunk_size)]

        with Pool(processes=self.num_workers) as p:
            p.map(self._start_worker, page_mapping)