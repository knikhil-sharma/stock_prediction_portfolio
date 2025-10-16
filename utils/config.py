# utils/config.py
import os

REDDIT_CREDS = {
    'client_id': os.getenv('REDDIT_CLIENT_ID'),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
    'user_agent': os.getenv('REDDIT_USER_AGENT')
}

NEWS_API_KEY = os.getenv('NEWS_API_KEY')