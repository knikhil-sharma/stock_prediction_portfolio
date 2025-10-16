from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import praw
from newsapi import NewsApiClient
import pandas as pd
from typing import Dict
import numpy as np

# Optional imports with fallbacks
try:
    from lime.lime_text import LimeTextExplainer
    from lime.explanation import Explanation
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    LimeTextExplainer = None
    Explanation = None 


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
            
        # Initialize LIME explainer if available
        if LIME_AVAILABLE:
            try:
                self.lime_explainer = LimeTextExplainer(class_names=['negative', 'positive'])
            except Exception as e:
                print(f"Warning: Could not initialize LIME: {e}")
                self.lime_explainer = None
        else:
            self.lime_explainer = None

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
        except Exception as e:
            print(f"Warning: News analysis failed: {e}")
            return pd.DataFrame({'vader': [0.0], 'source': ['demo']})

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
        except Exception as e:
            print(f"Warning: Reddit analysis failed: {e}")
            return pd.DataFrame({'vader': [0.0], 'source': ['demo']})

    def explain_sentiment(self, text: str):
        """Generate LIME explanation for sentiment prediction"""
        if self.lime_explainer is None:
            return None
            
        def predict_proba(texts):
            return np.array([[self.vader.polarity_scores(t)['compound'] if self.vader else 0.0] 
                       for t in texts])
    
        try:
            return self.lime_explainer.explain_instance(
                text,
                predict_proba,
                num_features=10
            )
        except Exception as e:
            print(f"Warning: Could not generate LIME explanation: {e}")
            return None

    def _get_finbert_score(self, text: str) -> float:
        if self.finbert is None:
            return 0.0
        try:
            result = self.finbert(text[:512])[0]
            return result['score'] if result['label'] == 'positive' else -result['score']
        except Exception:
            return 0.0
