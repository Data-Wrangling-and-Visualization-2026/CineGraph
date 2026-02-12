# CineGraph

## Project structure

```text
src/
│
├── backend/
│   ├── scraper/
│   │   ├── scraper.py        # Core scraping logic
│   │   └── utils.py          # Helper functions
│   │
│   ├── data/                 # Output data directory
│   │
│   ├── main.py               # Application entry point
│   ├── settings.py           # Configuration
│   ├── requirements.txt
│   └── Dockerfile
│
└── infra/
    └── docker-compose.yml    # Container orchestration
```

## How to run:

### Using docker
```bash
cd src/infra
docker compose up --build
```

### Manually

```bash
cd src/backend
python main.py
```

**Note**: on some systems selenium will require specific driver for the Chrome, so you will need replace:
(src/backend/scraper/scraper.py (40))
```python
service = Service(executable_path='/usr/bin/chromedriver')
```

with

```python
from webdriver_manager.chrome import ChromeDriverManager
service = Service(ChromeDriverManager().install())
```