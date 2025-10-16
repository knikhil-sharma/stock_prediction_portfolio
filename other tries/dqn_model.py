import numpy as np
from tensorflow.keras.models import load_model

# Load the trained DQN model
def load_dqn_model():
    return load_model("app/dqn_stock_model.h5")  # Ensure the correct path for your model file

# Preprocess user profile: Use the same 3 features as the model expects
def preprocess_user_profile(profile):
    # Example: Mapping the profile to match the model's 3 input features
    # Here we assume profile contains 'price', 'reddit_sentiment', and 'news_sentiment'
    return np.array([[
        profile['price'],              # Stock price
        profile['reddit_sentiment'],   # Sentiment from Reddit
        profile['news_sentiment']      # Sentiment from news
    ]])

# Predict the investment action (Buy, Hold, Sell)
def predict_investment(model, processed_input):
    prediction = model.predict(processed_input, verbose=0)
    return prediction
