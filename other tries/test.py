from data_sources.market_data import MarketDataFetcher
from models.rl_agent import DQNAgent
from utils.preprocess import FinancialPreprocessor, FeatureEngineer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def train_model():
    try:
        market = MarketDataFetcher()
        
        # Get validated tech stocks
        tech_stocks = market.get_sector_stocks("Information Technology")
        if not tech_stocks:
            raise ValueError("No tech stocks found in sector map")
            
        logging.info(f"Training on tech stocks: {tech_stocks[:3]}")
        
        # Fetch data with validation
        raw_data = market.get_historical_data(tech_stocks[:3])
        if raw_data.empty:
            raise ValueError("No historical data fetched after retries")
            
        # Validate columns
        required = ['Close', 'Volume', 'RSI', 'MACD']
        missing = [col for col in required if col not in raw_data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Preprocess data
        preprocessor = FinancialPreprocessor(
            numerical_features=required,
            categorical_features=[]
        )
        processed_data = preprocessor.transform(raw_data[required])
        
        # Initialize and train agent
        agent = DQNAgent(input_size=processed_data.shape[1], output_size=3)
        
        # Training loop with epsilon-greedy exploration
        for epoch in range(5):
            state = processed_data[0]
            for i in range(1, len(processed_data)):
                # Get action with exploration
                if np.random.rand() <= agent.epsilon:
                    action = np.random.randint(0, 3)
                else:
                    action = agent.model.predict(
                        state.reshape(1, -1), 
                        verbose=0
                    ).argmax()
                
                # Get next state and reward
                next_state = processed_data[i]
                reward = calculate_reward(state, next_state)
                done = i == len(processed_data)-1  # End of episode flag
                
                # Store experience and train
                agent.train(
                    [state], 
                    [action], 
                    [reward], 
                    [next_state], 
                    [done]
                )
                state = next_state
            
            logging.info(f"Epoch {epoch+1} completed. Epsilon: {agent.epsilon:.2f}")
            
        agent.model.save("models/rl_portfolio.h5")
        logging.info("Training completed successfully")

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

def calculate_reward(current_state, next_state):
    try:
        price_change = (next_state[0] - current_state[0]) / current_state[0]
        return price_change * 100
    except (IndexError, ZeroDivisionError):
        return 0

if __name__ == "__main__":
    train_model()
