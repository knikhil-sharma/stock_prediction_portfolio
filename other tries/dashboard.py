import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import logging
import tensorflow as tf
import streamlit as st

st.set_page_config(
    page_title="AI Portfolio Advisor",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Modern dark theme and card CSS
st.markdown("""
<style>
.stApp { background-color: #18191A; }
.metric-card {
    background: #23272F;
    border-radius: 12px;
    padding: 18px 18px 12px 18px;
    margin: 10px 0 10px 0;
    min-width: 140px;
    box-shadow: 0 2px 8px #0002;
}
.metric-label {color:#8b949e;font-size:1rem;}
.metric-value {color:#fff;font-size:1.3rem;font-weight:600;}
.metric-ticker {color:#fff;font-size:1.5rem;font-weight:700;text-align:center;margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)

from data_sources.market_data import MarketDataFetcher
from models.portfolio_opt import HybridPortfolioOptimizer as PortfolioOptimizer
from models.sentiment import SentimentAnalyzer
from models.rl_agent import DQNAgent
from utils.preprocess import FeatureEngineer
from models.xai import XAIExplainer

logging.basicConfig(level=logging.INFO)
MODEL_PATH = "models/high_confidence_rl_portfolio.h5"

@st.cache_resource
def initialize_components():
    market = MarketDataFetcher()
    sentiment = SentimentAnalyzer(
        st.secrets["api_keys"]["NEWS_API_KEY"],
        {
            'client_id': st.secrets["api_keys"]["REDDIT_CLIENT_ID"],
            'client_secret': st.secrets["api_keys"]["REDDIT_CLIENT_SECRET"],
            'user_agent': st.secrets["api_keys"]["REDDIT_USER_AGENT"]
        }
    )
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        if model.input_shape[1] != 10:
            raise ValueError("Model must have 10 input features")
        rl_agent = DQNAgent(input_size=10)
        rl_agent.model = model
        xai = XAIExplainer()
    except Exception as e:
        st.error(f"CRITICAL: {str(e)}")
        raise
    return market, sentiment, rl_agent, xai

market, sentiment_analyzer, rl_agent, xai = initialize_components()
risk_mapping = {'Low': 0.3, 'Medium': 0.5, 'High': 1.0}

@st.cache_data
def get_sector_sentiments_and_quotes(tickers):
    news_scores, reddit_scores = [], []
    quotes = market.get_realtime_quotes(tickers)
    for ticker in tickers:
        try:
            news = sentiment_analyzer.analyze_news_sentiment(ticker)
            reddit = sentiment_analyzer.analyze_reddit_sentiment(ticker)
            news_score = news['vader'].mean() if isinstance(news, pd.DataFrame) else 0
            reddit_score = reddit['vader'].mean() if isinstance(reddit, pd.DataFrame) else 0
            news_scores.append(news_score)
            reddit_scores.append(reddit_score)
        except Exception as e:
            logging.error(f"Sentiment error for {ticker}: {str(e)}")
            news_scores.append(0)
            reddit_scores.append(0)
    return news_scores, reddit_scores, quotes

def display_metrics_and_sentiment(ticker, quote, news_score, reddit_score):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-ticker">{ticker}</div>
        <div class="metric-label">Current Price</div>
        <div class="metric-value">${quote.get('price', 'N/A'):.2f}</div>
        <div class="metric-label">PE Ratio</div>
        <div class="metric-value">{quote.get('pe_ratio', 'N/A')}</div>
        <div class="metric-label">Volume</div>
        <div class="metric-value">{quote.get('volume', 0):,}</div>
        <div class="metric-label">News Sentiment</div>
        <div class="metric-value">{news_score:.2f}</div>
        <div class="metric-label">Reddit Sentiment</div>
        <div class="metric-value">{reddit_score:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

def explain_sentiment_paragraph():
    st.markdown("""
    <div style="background:#333;padding:14px;border-radius:8px;margin-top:12px;color:#fff;">
    <b>How to read sentiment scores?</b><br>
    <ul>
      <li>Scores range from -1 (negative) to +1 (positive)</li>
      <li>> 0.5: Strong positive</li>
      <li>0.1-0.5: Mild positive</li>
      <li>-0.1-0.1: Neutral</li>
      <li>< -0.1: Negative</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def display_rl_recommendation(action_probs):
    actions = ["Reduce Exposure", "Maintain Position", "Increase Exposure"]
    normalized_probs = tf.nn.softmax(action_probs).numpy()
    action_idx = np.argmax(normalized_probs)
    colors = ["#FF5252", "#FFD700", "#4CAF50"]
    st.markdown(f"""
    <div style="background-color: {colors[action_idx]}; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0;">{actions[action_idx]}</h2>
        <p style="color: white; margin: 5px 0 0 0;">Confidence: {normalized_probs[action_idx]*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    confidence_df = pd.DataFrame({
        'Action': actions,
        'Confidence (%)': normalized_probs * 100
    }).set_index('Action')
    st.write("Action Confidence Distribution:")
    st.bar_chart(confidence_df)

def plot_allocation(tickers, weights):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

def create_technical_indicators_chart(data):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(data.index, data['RSI'], label='RSI')
    ax.axhline(70, color='red', linestyle='--')
    ax.axhline(30, color='green', linestyle='--')
    ax.set_ylabel('RSI')
    st.pyplot(fig)

def explain_rl_decision(state, action_probs, context):
    try:
        background = np.random.randn(100, 10)
        shap_values = xai.explain_prediction(rl_agent.model, state, background=background)
        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 4))
        action_idx = np.argmax(action_probs)
        # Handle 3D or 2D SHAP output
        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
            shap.plots.bar(shap_values[0, action_idx], max_display=10, show=False)
        elif hasattr(shap_values, "values") and shap_values.values.ndim == 2:
            shap.plots.bar(shap_values[0], max_display=10, show=False)
        else:
            # fallback: show all features
            st.warning("Could not determine correct SHAP output shape. Showing all features.")
            shap.plots.bar(shap_values, max_display=10, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Explanation error: {str(e)}")
        st.markdown("""
        <div style="background:#222;padding:15px;border-radius:8px;color:#fff;">
        Key factors considered:
        <ul>
          <li>Technical indicators (RSI, MACD)</li>
          <li>Market volatility</li>
          <li>Investor sentiment</li>
          <li>Portfolio risk profile</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.title("AI Portfolio Advisor")
    with st.sidebar:
        st.header("Investor Profile")
        investment = st.number_input("Investment Amount ($)", 1000, 1000000, 10000)
        risk = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
        horizon = st.slider("Investment Horizon (years)", 1, 10, 5)
        analysis_type = st.radio("Analysis Type", ["Sector", "Stock"])
        query = st.text_input(f"Enter {analysis_type} Name")
    if query:
        if analysis_type == "Sector":
            analyze_sector(query, risk, investment, horizon)
        else:
            analyze_stock(query, risk, investment, horizon)

def analyze_sector(sector, risk, investment, horizon):
    try:
        with st.spinner(f"Analyzing {sector} sector..."):
            sector_stocks = market.get_sector_stocks(sector)
            if not sector_stocks:
                st.warning("No stocks found in this sector")
                return
            tickers = sector_stocks[:5]
            st.header(f"{sector} Sector Analysis")
            hist_data = market.get_historical_data(tickers, period="1y")
            returns = hist_data.pivot(index='Date', columns='Ticker', values='Close').pct_change().dropna()
            processed_data = FeatureEngineer().add_technical_indicators(hist_data)
            news_scores, reddit_scores, quotes = get_sector_sentiments_and_quotes(tickers)
            st.subheader("Sector Components: Key Metrics & Sentiment")
            cols = st.columns(5)
            for i, ticker in enumerate(tickers):
                with cols[i]:
                    display_metrics_and_sentiment(ticker, quotes.get(ticker, {}), news_scores[i], reddit_scores[i])
            explain_sentiment_paragraph()
            state = np.array([
                returns.mean().mean(),
                processed_data['RSI'].mean(),
                processed_data['MACD'].mean(),
                processed_data['Volume'].mean(),
                risk_mapping[risk],
                horizon,
                investment,
                np.mean(news_scores),
                np.mean(reddit_scores),
                returns.std().mean()
            ]).reshape(1, -1)
            st.subheader("Optimized Portfolio Allocation")
            optimizer = PortfolioOptimizer(risk)
            weights = optimizer.optimize(returns, state)
            plot_allocation(tickers, weights)
            action_probs = rl_agent.model.predict(state, verbose=0)[0]
            display_rl_recommendation(action_probs)
            with st.expander("Advanced Analysis"):
                explain_rl_decision(state, action_probs, sector)
    except Exception as e:
        logging.error(f"Sector analysis failed: {str(e)}")
        st.error(f"Sector analysis failed: {str(e)}")

def analyze_stock(ticker, risk, investment, horizon):
    try:
        with st.spinner(f"Analyzing {ticker}..."):
            hist_data = market.get_historical_data([ticker], period="1y")
            quotes = market.get_realtime_quotes([ticker])
            if hist_data.empty or not quotes:
                raise ValueError("No data available for this stock")
            processed_data = FeatureEngineer().add_technical_indicators(hist_data)
            if 'Date' in processed_data.columns:
                processed_data = processed_data.set_index('Date')
            news_sentiment = sentiment_analyzer.analyze_news_sentiment(ticker)
            reddit_sentiment = sentiment_analyzer.analyze_reddit_sentiment(ticker)
            avg_news = news_sentiment['vader'].mean() if isinstance(news_sentiment, pd.DataFrame) else 0
            avg_reddit = reddit_sentiment['vader'].mean() if isinstance(reddit_sentiment, pd.DataFrame) else 0
            st.subheader("Stock Key Metrics & Sentiment")
            display_metrics_and_sentiment(ticker, quotes[ticker], avg_news, avg_reddit)
            explain_sentiment_paragraph()
            state = np.array([
                processed_data['Close'].pct_change().mean(),
                processed_data['RSI'].iloc[-1],
                processed_data['MACD'].iloc[-1],
                processed_data['Volume'].iloc[-1],
                risk_mapping[risk],
                horizon,
                investment,
                avg_news,
                avg_reddit,
                processed_data['Close'].pct_change().std()
            ]).reshape(1, -1)
            action_probs = rl_agent.model.predict(state, verbose=0)[0]
            display_rl_recommendation(action_probs)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("Price History")
                st.line_chart(processed_data['Close'])
                st.subheader("Technical Indicators")
                create_technical_indicators_chart(processed_data)
            with st.expander("Advanced Analysis"):
                explain_rl_decision(state, action_probs, ticker)
    except Exception as e:
        logging.error(f"Stock analysis failed: {str(e)}")
        st.error(f"Stock analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
