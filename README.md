---
title: CineGraph
emoji: ⚡
colorFrom: green
colorTo: green
sdk: docker
pinned: false
license: mit
---

<a id="readme-top"></a>


<!-- PROJECT SHIELDS -->
<p align="center">
  <a href="https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/Data-Wrangling-and-Visualization-2026/CineGraph.svg?style=for-the-badge" />
  </a>
  <a href="https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph/network/members">
    <img src="https://img.shields.io/github/forks/Data-Wrangling-and-Visualization-2026/CineGraph.svg?style=for-the-badge" />
  </a>
  <a href="https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph/stargazers">
    <img src="https://img.shields.io/github/stars/Data-Wrangling-and-Visualization-2026/CineGraph.svg?style=for-the-badge" />
  </a>
  <a href="https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph/issues">
    <img src="https://img.shields.io/github/issues/Data-Wrangling-and-Visualization-2026/CineGraph.svg?style=for-the-badge" />
  </a>
  <a href="https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph/blob/master/LICENSE.txt">
    <img src="https://img.shields.io/github/license/Data-Wrangling-and-Visualization-2026/CineGraph.svg?style=for-the-badge" />
  </a>
</p>



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph">
    <!-- Replace this with your actual project logo -->
    <!-- <img src="https://cdn-icons-png.flaticon.com/512/3172/3172554.png" alt="Logo" width="80" height="80"> -->
  </a>

<h3 align="center">CineGraph</h3>

  <p align="center">
    A data-driven web application that processes raw subtitles from 40,000 movies to generate interactive "Emotional Seismographs"
    <br />
    <a href="https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph">View Demo</a>
    &middot;
    <a href="https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph/issues">Report Bug</a>
    &middot;
    <a href="https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
        <li><a href="#project-structure">Project Structure</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#data-pipeline">Data pipeline</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

We are building **CineGraph**, a data-driven web application that processes raw subtitles from 40,000 movies to generate interactive ”Emotional Seismographs”.

Users can inspect emotionally close movies and select a film to view its sentiment arc (consisting of 6 main [emotions](https://www.verywellmind.com/an-overview-of-the-types-of-emotions-4163976)) over time. The system combines a massive scraping pipeline with NLP analysis and advanced clustering techniques to reveal the hidden structure of storytelling.

*Main subtitle source:* [SubsLikeScript](https://subslikescript.com/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python.js]][Python-url]
* [![LangChain][LangChain.js]][LangChain-url]
* [![PostgreSQL][PostgreSQL.js]][PostgreSQL-url]
* [![Selenium][Selenium.js]][Selenium-url]
* [![Docker][Docker.js]][Docker-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Project Structure

Here is an overview of the core structure and module responsibilities:

```text
CINEGRAPH/
├── .venv/                     # Virtual environment
├── src/
│   ├── backend/               # Main Backend application
│   │   ├── api/               # API endpoints and routing
│   │   ├── clustering/        # Graph creation and dataset processing algorithms
│   │   │   ├── graph_creator.py
│   │   │   └── utils.py
│   │   ├── data/              # Local data/storage dumps
│   │   ├── db/                # Database configurations, models, and migrations
│   │   │   ├── base.py
│   │   │   └── session.py
│   │   ├── emotion_analysis/  # NLP analysis models, embeddings, and weights
│   │   │   └── model.py
│   │   ├── experiments/       # Sandbox for testing scripts and models
│   │   ├── preprocessing/     # LangChain agents for data cleaning
│   │   │   └── preprocessing_agent.py
│   │   ├── scraping/          # Selenium pipeline for pulling raw subtitles
│   │   │   ├── scraper.py
│   │   │   └── utils.py
│   │   ├── services/          # Core business logic and external integrations
│   │   ├── .env               # Backend specific environment variables
│   │   ├── dockerfile         # Dockerfile for backend service
│   │   ├── main.py            # Application entry point (Pipeline + FastAPI)
│   │   ├── requirements.txt   # Backend-specific dependencies
│   │   └── settings.py        # App configuration settings
│   └── infra/                 # Infrastructure and Orchestration
│       ├── .env               # Infrastructure specific environment variables
│       └── docker-compose.yml # Docker Compose to spin up app and PostgresDB
├── .gitignore
├── README.md                  # Project documentation
└── requirements_full.txt      # Global project dependencies
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

Ensure you have the following installed on your machine:
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* [Python 3.11+](https://www.python.org/downloads/) (if running natively)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph.git
   cd CineGraph
   ```

2. **Environment Variables**: Create your `.env` files. You will need to populate both `src/backend/.env` and `src/infra/.env` with the following template:
   ```env
   # List of proxies for scraping
   IP_1=
   IP_2=
   IP_N=
   PROXY_PORT=3128

   DB_URL="postgresql+asyncpg://{user}:{password}@{host}:{port}/movies"

   API_PORT=5555
   FRONT_PORT=5173
   ```

3. Spin up the infrastructure and application using Docker Compose:
   ```sh
   cd src/infra
   docker-compose up --build
   ```

4. *(Optional)* If you wish to run the app natively without Docker:
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements_full.txt
   ```

    For the frontend:
    ```sh
    cd src/frontend
    npm install
    npm run dev
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Data pipeline

* **Scraping**: The Selenium scraper (`src/backend/scraping/scraper.py`) utilizes the proxy list from your `.env` to scrape scripts from *subslikescript.com* extending request rate limitation (of course, with all the grace for the source website). The pipeline will also work without proxy servers.
* **Preprocessing**: Raw subtitles are passed to the LangChain agent (`src/backend/preprocessing/preprocessing_agent.py`) to clean the original text.
* **Emotion Analysis**: The NLP model (`src/backend/emotion_analysis/model.py`) evaluates the emotional trajectory of the subtitle windows. The model outputs embedding with 6 emotion intensities.
* **Clustering & Graphing**: Graph modules process the data and saves it in tree-based format to `PostgreSQL`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Build Selenium scraping pipeline for SubsLikeScript
- [x] Integrate LangChain for data preprocessing
- [x] Implement Emotion Analysis with NLP model
- [x] Integrate PostgreSQL
- [x] Complete graph clustering algorithm
- [ ] Design & Implement main API
- [ ] Build interactive web frontend for the Graph representation
- [ ] Build interactive Web Frontend for the "Emotional Seismographs"

See the [open issues](https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feat/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feat/AmazingFeature`)
5. Open a Pull Request

Additionally, it is highly recommended to follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) style guide.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Maybe, we will add it later...

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [SubsLikeScript](https://subslikescript.com/) - Main Subtitle Source
* [LangChain](https://python.langchain.com/) - LLM Application Framework

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Data-Wrangling-and-Visualization-2026/CineGraph.svg?style=for-the-badge
[contributors-url]: https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/Data-Wrangling-and-Visualization-2026/CineGraph.svg?style=for-the-badge
[forks-url]: https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph/network/members

[stars-shield]: https://img.shields.io/github/stars/Data-Wrangling-and-Visualization-2026/CineGraph.svg?style=for-the-badge
[stars-url]: https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph/stargazers

[issues-shield]: https://img.shields.io/github/issues/Data-Wrangling-and-Visualization-2026/CineGraph.svg?style=for-the-badge
[issues-url]: https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph/issues

[license-shield]: https://img.shields.io/github/license/Data-Wrangling-and-Visualization-2026/CineGraph.svg?style=for-the-badge
[license-url]: https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph/blob/master/LICENSE.txt

<!-- Tech Stack Badges -->
<!-- SHIELDS -->
[issues-shield]: https://img.shields.io/github/issues/Data-Wrangling-and-Visualization-2026/CineGraph.svg?style=for-the-badge
[issues-url]: https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph/issues

[license-shield]: https://img.shields.io/github/license/Data-Wrangling-and-Visualization-2026/CineGraph.svg?style=for-the-badge
[license-url]: https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph/blob/master/LICENSE

<!-- Tech Stack -->

[Python-url]: https://www.python.org/

[LangChain.js]: https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white
[LangChain-url]: https://python.langchain.com/

[PostgreSQL.js]: https://img.shields.io/badge/postgresql-4169e1?style=for-the-badge&logo=postgresql&logoColor=white
[PostgreSQL-url]: https://www.postgresql.org/

[Selenium.js]: https://img.shields.io/badge/-selenium-43B02A?style=for-the-badge&logo=selenium&logoColor=white
[Selenium-url]: https://www.selenium.dev/

[Docker.js]: https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white
[Docker-url]: https://www.docker.com/

[Python.js]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54