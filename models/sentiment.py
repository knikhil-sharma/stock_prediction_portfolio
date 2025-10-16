from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import praw
from newsapi import NewsApiClient
import pandas as pd
from typing import Dict
import numpy as np
from lime.lime_text import LimeTextExplainer
from lime.explanation import Explanation 


class SentimentAnalyzer:
    """Multi-source sentiment analysis with XAI integration"""
    
    def __init__(self, news_api_key: str, reddit_config: Dict):
        self.vader = SentimentIntensityAnalyzer()
        self.finbert = pipeline("text-classification", 
                              model="yiyanghkust/finbert-tone")
        self.news_client = NewsApiClient(api_key=news_api_key)
        self.reddit = praw.Reddit(**reddit_config)
        self.lime_explainer = LimeTextExplainer(class_names=['negative', 'positive'])

    def analyze_news_sentiment(self, query: str) -> pd.DataFrame:
        """Analyze news sentiment with source tracking"""
        articles = self.news_client.get_everything(
            q=query,
            language='en',
            page_size=50
        )['articles']
        
        return pd.DataFrame([{
            'source': a['source']['name'],
            'vader': self.vader.polarity_scores(f"{a['title']}. {a.get('description','')}")['compound'],
            'finbert': self._get_finbert_score(f"{a['title']}. {a.get('description','')}"),
            'url': a['url']
        } for a in articles])

    def analyze_reddit_sentiment(self, query: str) -> pd.DataFrame:
        """Analyze Reddit sentiment with temporal features"""
        posts = self.reddit.subreddit('stocks').search(query, limit=100)
        
        return pd.DataFrame([{
            'created_utc': post.created_utc,
            'vader': self.vader.polarity_scores(f"{post.title} {post.selftext}")['compound'],
            'finbert': self._get_finbert_score(f"{post.title} {post.selftext}"),
            'url': f"https://reddit.com{post.permalink}"
        } for post in posts])

    def explain_sentiment(self, text: str) -> Explanation:
        """Generate LIME explanation for sentiment prediction"""
        def predict_proba(texts):
            return np.array([[self.vader.polarity_scores(t)['compound']] 
                       for t in texts])
    
        return self.lime_explainer.explain_instance(
        text,
        predict_proba,
        num_features=10
    )

    def _get_finbert_score(self, text: str) -> float:
        result = self.finbert(text[:512])[0]
        return result['score'] if result['label'] == 'positive' else -result['score']
