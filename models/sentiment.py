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
        try:
            self.vader = SentimentIntensityAnalyzer()
        except Exception as e:
            print(f"Warning: Could not initialize VADER: {e}")
            self.vader = None
            
        try:
            self.finbert = pipeline("text-classification", 
                                  model="yiyanghkust/finbert-tone")
        except Exception as e:
            print(f"Warning: Could not initialize FinBERT: {e}")
            self.finbert = None
            
        try:
            self.news_client = NewsApiClient(api_key=news_api_key)
        except Exception as e:
            print(f"Warning: Could not initialize News API: {e}")
            self.news_client = None
            
        try:
            self.reddit = praw.Reddit(**reddit_config)
        except Exception as e:
            print(f"Warning: Could not initialize Reddit: {e}")
            self.reddit = None
        self.lime_explainer = LimeTextExplainer(class_names=['negative', 'positive'])

    def analyze_news_sentiment(self, query: str) -> pd.DataFrame:
        """Analyze news sentiment with source tracking"""
        if self.news_client is None:
            return pd.DataFrame({'vader': [0.0], 'source': ['demo']})
            
        try:
            articles = self.news_client.get_everything(
                q=query,
                language='en',
                page_size=50
            )['articles']
        
        return pd.DataFrame([{
            'source': a['source']['name'],
            'vader': self.vader.polarity_scores(f"{a['title']}. {a.get('description','')}")['compound'] if self.vader else 0.0,
            'finbert': self._get_finbert_score(f"{a['title']}. {a.get('description','')}"),
            'url': a['url']
        } for a in articles])

    def analyze_reddit_sentiment(self, query: str) -> pd.DataFrame:
        """Analyze Reddit sentiment with temporal features"""
        if self.reddit is None:
            return pd.DataFrame({'vader': [0.0], 'source': ['demo']})
            
        try:
            posts = list(self.reddit.subreddit('stocks').search(query, limit=100))
        
        return pd.DataFrame([{
            'created_utc': post.created_utc,
            'vader': self.vader.polarity_scores(f"{post.title} {post.selftext}")['compound'] if self.vader else 0.0,
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
        if self.finbert is None:
            return 0.0
        try:
            result = self.finbert(text[:512])[0]
            return result['score'] if result['label'] == 'positive' else -result['score']
        except Exception:
            return 0.0
