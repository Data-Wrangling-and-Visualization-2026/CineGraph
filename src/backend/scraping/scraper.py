import os
import re
from multiprocessing import Pool
from pathlib import Path

from scraping.utils import create_data_folder, get_next_proxy
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from settings import BASE_DIR, settings
from webdriver_manager.chrome import ChromeDriverManager


class Scraper:
    """
    Scraper for subtitles
    """
    def __init__(self) -> None:
        # Verify existance of data folder
        create_data_folder()

        self.BASE_URL = settings.scraper.url # Base url for scrapping
        self.output_folder = settings.scraper.output_folder # Default is /data
        self.delay = settings.scraper.delay # Delay between scraping two pages
        self.num_workers = settings.scraper.num_workers


    def _setup_driver(self, proxy: str | None = None) -> WebDriver:
        """
        Setups driver for worker.

        Args:
            proxy (str | None, optional): additional proxy server. Defaults to None.

        Returns:
            WebDriver: driver with selected options
        """
        chrome_opts = Options()

        if settings.scraper.headless:
            chrome_opts.add_argument('--headless') # disable Chrome UI

        # Optimization for Docker
        chrome_opts.add_argument('--disable-gpu')
        chrome_opts.add_argument('--no-sandbox')
        chrome_opts.add_argument("--disable-dev-shm-usage")

        # Simulates user
        chrome_opts.add_argument(
            'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' \
            '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )

        if proxy:
            print(proxy)
            chrome_opts.add_argument(f'--proxy-server=http://{proxy}')

        # Uncomment second if run manually. Keep, if run inside Docker
        # This installs or collects already existing Chrome driver for selenium
        root = BASE_DIR.cwd()
        if root == Path('app'):
            service = Service(executable_path='/usr/bin/chromedriver')
        else:
            service = Service(ChromeDriverManager().install())

        driver = webdriver.Chrome(service=service, options=chrome_opts)

        return driver


    def _separate_year_and_title(self, text: str) -> tuple[str, str]:
        """
        Splits "Title (year)" into separate values. If year is missing, then 1800 is returned.

        Args:
            text (str): title with year

        Returns:
            tuple[str, str]: (title, year)
        """
        match = re.search(r'(.*)\s\((\d{4})\)$', text)
        if match:
            return match.group(1).strip(), match.group(2)
        return text.strip(), '1800'


    def _get_movie_links(self, driver: WebDriver, page: int) -> list[dict[str, str]]:
        """
        Parses the page with list of links to concrete subtitles

        Args:
            driver (WebDriver): worker's driver
            page (int): page id (number)

        Returns:
            list[dict[str, str]]: List of dicts with movie metadata (title, year, url)
        """
        print(f'Collecting links from page #{page}')

        movie_metadata = []
        url = f'{self.BASE_URL}/movies?page={page}'
        driver.get(url)

        try:
            # Waits for couple of seconds to render the whole list
            WebDriverWait(driver, self.delay).until(
                EC.presence_of_element_located(
                    (By.CLASS_NAME, 'scripts-list')
                )
            )

            # Selects list elements
            links = driver.find_elements(By.CSS_SELECTOR, '.scripts-list li a')

            # Constructs movies metadata
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
        """
        Saves subtitles in output folder with the name 'title_year.txt'

        Args:
            movie (dict[str, str]): movie's metadata
            text (str): subtitles
        """
        path = os.path.join(
            self.output_folder,
            f'{movie["title"]}_{movie["year"]}.txt'.replace(' ', '_'),
        )

        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f'{movie["title"]} {movie["year"]} saved!')


    def _scrape_subtitles(self, driver: WebDriver, movie: dict) -> str:
        """
        Scrapes subtitles for the selected movie

        Args:
            driver (WebDriver): worker's driver
            movie (dict): movie's metadata

        Returns:
            str: subtitles
        """
        print(f'Scraping {movie["title"]} {movie["year"]}')

        # load url
        driver.get(movie['url'])

        full_script = ''

        try:
            # Wait until all subtitles are present
            WebDriverWait(driver, self.delay).until(
                EC.presence_of_element_located((By.ID, 'cue-app'))
            )

            # Select only text
            lines = driver.find_elements(By.CLASS_NAME, 'cue-line')

            if not lines:
                print(f'No subtitles for {movie["title"]} {movie["year"]}')

            lines = [line.text for line in lines]
            full_script = '\n'.join(lines)

            self._save(movie, full_script)
        except Exception as e:
            print(f'Error scraping detail page for {movie["title"]}: {e}')

        return full_script


    def _start_worker(self, page_range: list[int], proxy: str | None, worker_id: int = 0) -> None:
        """
        Starts worker.

        Args:
            page_range (list[int]): list of page numbers to be scraped
            proxy (str | None): optional proxy
            worker_id (int, optional): id. Defaults to 0.
        """
        driver = self._setup_driver(proxy)

        try:
            all_movies = []

            # Select all links from page with movies -> scrap & save subtitles
            for page in page_range:
                movies_on_page = self._get_movie_links(driver, page)
                for movie in movies_on_page:
                    self._scrape_subtitles(driver, movie)

                all_movies.extend(movies_on_page)

        except Exception as e:
            print(e)

        finally:
            driver.quit() # Most important part. Do not change!
            print(f'Worker №{worker_id} finished scraping!')


    def start_scraping(self) -> None:
        """
        Starts scrapping
        """
        start_page = 1
        end_page = 2029
        offset = settings.scraper.offset

        # Map pages to workers
        all_pages = list(range(start_page, end_page + 1))
        chunk_size = (len(all_pages) + self.num_workers - 1) // self.num_workers
        page_mapping = [
            all_pages[i + offset : i + offset + chunk_size]
            for i in range(0, len(all_pages), chunk_size)
        ]

        # Map optional proxy servers. If no present -> populate with Nones
        proxies = get_next_proxy()
        proxies = [
            next(proxies) for _ in range(self.num_workers)
        ]

        # Collect worker's args
        worker_args = [
            (page_map, proxy, i + 1)
            for i, (page_map, proxy) in enumerate(zip(page_mapping, proxies))
        ]

        # Each worker - separate process
        with Pool(processes=self.num_workers) as p:
            p.starmap(self._start_worker, worker_args)