from asyncio import run

from clustering.graph_creator import GraphCreator
from emotion_analysis.model import EmotionAnalyzer
from preprocessing.preprocessing_agent import PreprocessingAgent
from scraping.scraper import Scraper


def pipeline() -> None:
    Scraper().start_scraping()
    PreprocessingAgent().start_preprocessing()
    EmotionAnalyzer().analyze_data()
    g = GraphCreator()
    run(g.construct_graph())


if __name__ == '__main__':
    pipeline()
