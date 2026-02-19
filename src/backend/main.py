# from scraping.scraper import Scraper
# from emotion_analysis.model import EmotionAnalyzer
from preprocessing.preprocessing_agent import PreprocessingAgent

if __name__ == '__main__':
    # EmotionAnalyzer().analyze_data()
    # Scraper().start_scraping()
    PreprocessingAgent().start_preprocessing()