# data_sources/news_client.py
from newsapi import NewsApiClient

class NewsAnalyzer:
    def __init__(self, api_key: str):
        self.newsapi = NewsApiClient(api_key=api_key)
        
    def get_news(self, query: str):
        try:
            return self.newsapi.get_everything(
                q=query,
                language='en',
                page_size=50
            )['articles']
        except Exception as e:
            print(f"News API error: {str(e)}")
            return []
