from asyncio import run

from clustering.graph_creator import GraphCreator
from emotion_analysis.model import EmotionAnalyzer
from preprocessing.preprocessing_agent import PreprocessingAgent
from scraping.scraper import Scraper


def pipeline() -> None:
    """
    Temporary starts the pipeline:
        1. Scraping -> data is placed into './data/'

        2. Preprocessing -> agent places ready data
           to '.preprocessing/ready_data/'

        3. Sentiment analysis -> files with embeddings
           are stored in './emotion_analysis/embeddings'

        4. Graph construction & db population -> clustering
           + dumping into db
    """
    Scraper().start_scraping()
    PreprocessingAgent().start_preprocessing()
    EmotionAnalyzer().analyze_data()
    g = GraphCreator()
    run(g.construct_graph()) # run() is used because construct_graph is async


if __name__ == '__main__':
    pipeline()
