from data_sources.market_data import MarketDataFetcher
from models.rl_agent import DQNAgent
from models.sentiment import SentimentAnalyzer
from utils.preprocess import FinancialPreprocessor, FeatureEngineer
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from collections import deque
import time
import requests
import tomli 

logging.basicConfig(level=logging.INFO)

# Load secrets from TOML file
with open(".streamlit\secrets.toml", "rb") as f:
    secrets = tomli.load(f)

class RateLimitedMarketDataFetcher(MarketDataFetcher):
    """Adds rate limiting and sentiment integration"""
    
    def __init__(self):
        super().__init__()
        self.sentiment_analyzer = SentimentAnalyzer(
            news_api_key=secrets["api_keys"]["NEWS_API_KEY"],
            reddit_config={
                'client_id': secrets["api_keys"]["REDDIT_CLIENT_ID"],
                'client_secret': secrets["api_keys"]["REDDIT_CLIENT_SECRET"],
                'user_agent': secrets["api_keys"]["REDDIT_USER_AGENT"]
            }
        )
        self.last_request_time = 0
        self.request_interval = 1.2  # 1.2 seconds between requests
        self.max_retries = 3
        self.retry_delay = 5

    def get_historical_data(self, tickers: list, period: str = "6mo") -> pd.DataFrame:
        """Fetch data with rate limiting and retries"""
        price_data = super().get_historical_data(tickers, period)
        
        try:
            sentiment_data = self._get_sentiment_with_retry(tickers, price_data.index)
        except Exception as e:
            logging.error(f"Sentiment failed: {str(e)} - continuing without")
            sentiment_data = pd.DataFrame(index=price_data.index)

        full_data = price_data.merge(sentiment_data, 
                                   left_index=True, right_index=True,
                                   how='left').fillna(0)
        
        return FeatureEngineer().add_technical_indicators(full_data)

    def _get_sentiment_with_retry(self, tickers, dates):
        """Get sentiment with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return self._get_sentiment_scores(tickers, dates)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait = self.retry_delay * (2 ** attempt)
                    logging.warning(f"Rate limited. Retrying in {wait}s")
                    time.sleep(wait)
                    continue
                raise
        return pd.DataFrame()

    def _get_sentiment_scores(self, tickers: list, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Safe sentiment scoring with rate limits"""
        scores = []
        for ticker in tickers:
            time.sleep(self.request_interval)
            
            try:
                news = self._get_sentiment(ticker, 'news')
                reddit = self._get_sentiment(ticker, 'reddit')
                
                combined = pd.DataFrame({
                    f'{ticker}_News': news['vader'],
                    f'{ticker}_Reddit': reddit['vader']
                })
                scores.append(combined)
            except Exception as e:
                logging.error(f"Sentiment failed for {ticker}: {str(e)}")
                continue
                
        if scores:
            return pd.concat(scores, axis=1).reindex(dates).fillna(0)
        return pd.DataFrame(index=dates)

    def _get_sentiment(self, ticker, source):
        """Get sentiment with rate limiting"""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        
        self.last_request_time = time.time()
        if source == 'news':
            return self.sentiment_analyzer.analyze_news_sentiment(ticker)
        return self.sentiment_analyzer.analyze_reddit_sentiment(ticker)

class HighConfidenceDQNAgent(DQNAgent):
    """Enhanced agent with confidence monitoring"""
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.target_model = self._build_model()
        self.update_target_every = 100
        self.step_count = 0
        self.memory = deque(maxlen=2000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def _build_model(self):
        """Deeper network with regularization"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', 
                                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                input_shape=(self.input_size,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.output_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
                    loss='mse',
                    metrics=['accuracy'])
        return model

    def train(self, states, actions, rewards, next_states, dones):
        """Experience replay with target network"""
        self.memory.extend(zip(states, actions, rewards, next_states, dones))
        
        if len(self.memory) < self.batch_size:
            return

        minibatch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = np.array([self.memory[i][0] for i in minibatch])
        actions = np.array([self.memory[i][1] for i in minibatch])
        rewards = np.array([self.memory[i][2] for i in minibatch])
        next_states = np.array([self.memory[i][3] for i in minibatch])
        dones = np.array([self.memory[i][4] for i in minibatch])

        target = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)
        
        target[np.arange(len(target)), actions] = rewards + self.gamma * np.max(target_next, axis=1) * ~dones
        self.model.fit(states, target, verbose=0)

        if self.step_count % self.update_target_every == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        self.step_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_model():
    try:
        market = RateLimitedMarketDataFetcher()
        tech_stocks = market.get_sector_stocks("Information Technology")[:3]
        logging.info(f"Training on stocks: {tech_stocks}")
        
        raw_data = market.get_historical_data(tech_stocks)
        if raw_data.empty:
            raise ValueError("No data fetched")
            
        required = ['Close', 'Volume', 'RSI', 'MACD'] + \
                  [f'{t}_News' for t in tech_stocks] + \
                  [f'{t}_Reddit' for t in tech_stocks]
        raw_data = raw_data[required].dropna()

        preprocessor = FinancialPreprocessor(
            numerical_features=required,
            categorical_features=[]
        )
        processed_data = preprocessor.transform(raw_data)
        
        agent = HighConfidenceDQNAgent(input_size=processed_data.shape[1], output_size=3)
        confidence_history = []
        
        for epoch in range(20):
            state = processed_data[0]
            total_reward = 0
            correct = 0
            
            for i in range(1, len(processed_data)):
                if np.random.rand() < agent.epsilon:
                    action = np.random.randint(3)
                else:
                    q_values = agent.model.predict(state.reshape(1, -1), verbose=0)[0]
                    action = np.argmax(q_values)
                    confidence = tf.nn.softmax(q_values)[action]
                    if confidence > 0.9:
                        correct += 1
                
                next_state = processed_data[i]
                reward = calculate_reward(state, next_state, action)
                agent.train([state], [action], [reward], [next_state], [i == len(processed_data)-1])
                state = next_state
                total_reward += reward
            
            accuracy = correct / (len(processed_data)-1)
            confidence_history.append(accuracy)
            logging.info(f"Epoch {epoch+1} | Accuracy: {accuracy:.2%} | Reward: {total_reward:.2f}")
            
            if accuracy >= 0.9:
                logging.info("High confidence achieved")
                break
        
        agent.model.save("models/confident_rl_portfolio.h5")

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

def calculate_reward(current_state, next_state, action):
    price_change = (next_state[0] - current_state[0]) / current_state[0]
    direction = 1 if action == 2 else -1 if action == 0 else 0
    return price_change * 100 + direction * 10

if __name__ == "__main__":
    train_model()
