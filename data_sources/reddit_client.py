# data_sources/reddit_client.py
import praw

class RedditClient:
    def __init__(self, config):
        self.reddit = praw.Reddit(
            client_id=config['client_id'],
            client_secret=config['client_secret'],
            user_agent=config['user_agent']
        )

    def get_posts(self, query, limit=100):
        return self.reddit.subreddit('stocks').search(query, limit=limit)
