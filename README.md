# 🎬 CineGraph

CineGraph is a data-driven web application that processes raw subtitles
from **45,000+ movies** to generate interactive **Emotional
Seismographs**.

Users can: - Select a movie - Visualize its sentiment arc over time -
Zoom into emotional peaks - Identify the exact phrases (lemmas) driving
emotional context

The system combines a large-scale scraping pipeline with context-aware
NLP analysis to reveal the hidden structure of storytelling.

**Main subtitle source:** https://subslikescript.com/

------------------------------------------------------------------------

# 🏗 Architecture Overview

CineGraph is built as a modular backend system with:

-   Scraping Layer → Collects subtitles\
-   Preprocessing Layer → Cleans and structures text\
-   Emotion Analysis Engine → Generates embeddings & sentiment scores\
-   Database Layer → Stores movies, subtitles, embeddings, metadata\
-   API Layer → Serves processed data\
-   Clustering & Graph Module → Builds narrative graphs\
-   Dockerized Infrastructure → Ensures reproducible environment

------------------------------------------------------------------------

# 🛠 Tech Stack

-   Python --- Core backend language\
-   LangChain -- NLP orchestration & embedding workflows\
-   PostgreSQL -- Primary relational database\
-   Selenium -- Subtitle scraping\
-   Docker & Docker Compose -- Containerization\
-   AsyncPG -- Async PostgreSQL driver

------------------------------------------------------------------------

# 📂 Project Structure

    CINEGRAPH/
    │
    ├── .venv/                     # Local virtual environment
    │
    ├── src/
    │   ├── backend/
    │   │
    │   │   ├── api/               # FastAPI routes / endpoints
    │   │   │
    │   │   ├── clustering/        # Graph building & movie clustering logic
    │   │   │   ├── balance.json
    │   │   │   ├── graph_creator.py
    │   │   │   ├── unbalanced.json
    │   │   │   └── utils.py
    │   │   │
    │   │   ├── data/              # Data storage helpers
    │   │   │
    │   │   ├── db/                # Database layer
    │   │   │   ├── migrations/    # DB migration scripts
    │   │   │   ├── models/        # SQLAlchemy models
    │   │   │   ├── repositories/  # Repository pattern abstraction
    │   │   │   ├── base.py        # Base model config
    │   │   │   └── session.py     # Async DB session management
    │   │   │
    │   │   ├── emotion_analysis/  # NLP & embedding pipeline
    │   │   │   ├── embeddings/    # Embedding generation
    │   │   │   ├── weights/       # Model weights
    │   │   │   ├── model.py       # Sentiment model logic
    │   │   │   └── experiments/   # Experimental NLP features
    │   │   │
    │   │   ├── preprocessing/     # Subtitle cleaning & normalization
    │   │   │   ├── raw_data/      # Raw subtitle dumps
    │   │   │   ├── passthrough.py
    │   │   │   └── preprocessing_agent.py
    │   │   │
    │   │   ├── scraping/          # Selenium-based scraping pipeline
    │   │   │   ├── scraper.py
    │   │   │   └── utils.py
    │   │   │
    │   │   ├── services/          # Business logic layer
    │   │   │
    │   │   ├── main.py            # Application entry point
    │   │   ├── settings.py        # Environment & config management
    │   │   ├── dockerfile         # Backend container config
    │   │   ├── requirements.txt   # Python dependencies
    │   │   └── .env               # Backend environment config
    │
    ├── infra/
    │   ├── docker-compose.yml     # Multi-container setup
    │   └── .env                   # Infrastructure-level env variables
    │
    ├── .gitignore
    ├── README.md
    └── requirements_full.txt

------------------------------------------------------------------------

# 🔐 Environment Configuration

Each `.env` file must include:

``` env
# List of proxies for scraping
IP_1=
IP_2=
IP_N=
PROXY_PORT=3128

# DB Related things
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=movies

DB_URL="postgresql+asyncpg://user:password@localhost:5432/movies"
```

### Notes

-   Multiple proxy IPs improve scraping resilience.
-   `DB_URL` uses asyncpg driver.
-   Ensure PostgreSQL is running before starting the backend.

------------------------------------------------------------------------

# 🐳 Running with Docker (Recommended)

## Build & Start Containers

    docker-compose up --build

## Stop Containers

    docker-compose down

------------------------------------------------------------------------

# 🧪 Running Locally (Without Docker)

## Create Virtual Environment

    python -m venv .venv
    source .venv/bin/activate   # Linux / Mac

## Install Dependencies

    pip install -r requirements.txt

## Run Backend

    uvicorn src.backend.main:app --reload

------------------------------------------------------------------------

# 🔄 Data Pipeline Flow

Scraping (Selenium) ↓ Raw Subtitle Storage ↓ Preprocessing (Cleaning +
Lemmatization) ↓ Embedding Generation (LangChain) ↓ Emotion Scoring
Model ↓ Database Storage ↓ Graph & Clustering ↓ API → Frontend
Visualization

------------------------------------------------------------------------

# 📊 Emotional Seismograph Concept

For each movie:

1.  Subtitles are segmented by time.
2.  Each segment is embedded using contextual embeddings.
3.  Sentiment intensity is computed.
4.  Scores are plotted over time.
5.  Peaks are linked to specific lemmas driving emotional weight.

This transforms raw subtitle text into a narrative emotion curve.

------------------------------------------------------------------------

# 🚀 Future Improvements

-   Frontend integration (React / D3.js for visualization)
-   Caching layer (Redis)
-   Horizontal scraping workers
-   Vector database integration
-   Movie recommendation based on emotional similarity

------------------------------------------------------------------------

# 👥 Project Vision

CineGraph aims to answer:

> What does emotion look like over time in storytelling?

By converting dialogue into structured emotional signals, we uncover
narrative patterns hidden in plain text.