
# from emotion_analysis.model import EmotionAnalyzer

# from preprocessing.preprocessing_agent import PreprocessingAgent
from asyncio import run

from clustering.graph_creator import GraphCreator

if __name__ == '__main__':
    # EmotionAnalyzer().analyze_data()
    # PreprocessingAgent().start_preprocessing()
    # Scraper().start_scraping()
    # PreprocessingAgent().start_preprocessing()
    g = GraphCreator()
    run(g.construct_graph())
