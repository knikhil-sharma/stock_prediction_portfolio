import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import logging
from utils.preprocess import FeatureEngineer
import time

class MarketDataFetcher:
    """Enhanced market data fetcher with robust error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sector_map = self._get_sp500_sectors() or self._get_fallback_sectors()
        
    def _get_sp500_sectors(self) -> Dict[str, List[str]]:
        """Scrape S&P 500 constituents with retry logic"""
        for _ in range(3):
            try:
                url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                response = requests.get(url, timeout=15)
                soup = BeautifulSoup(response.text, 'html.parser')
                return self._parse_sectors(soup)
            except Exception as e:
                self.logger.warning(f"Retrying sector mapping: {str(e)}")
                time.sleep(2)
        return {}

    def _get_fallback_sectors(self) -> Dict[str, List[str]]:
        """Fallback sector mapping with popular stocks when Wikipedia fails"""
        return {
            'Information Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'ORCL', 'CRM', 'ADBE'],
            'Health Care': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR'],
            'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'KMI', 'PXD', 'MPC'],
            'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX'],
            'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS'],
            'Industrials': ['BA', 'CAT', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'MMM'],
            'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'CHTR'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'SPG', 'O'],
            'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'PPG', 'NEM']
        }

    def _parse_sectors(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Parse Wikipedia table with validation"""
        sector_map = {}
        try:
            table = soup.find('table', {'class': 'wikitable'})
            if not table:
                return {}
                
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) >= 5:
                    ticker = cols[0].text.strip()
                    sector = cols[2].text.strip()
                    if sector and ticker:
                        sector = self._normalize_sector_name(sector)
                        if self._is_valid_ticker(ticker):
                            sector_map.setdefault(sector, []).append(ticker)
            return sector_map
        except Exception as e:
            self.logger.error(f"Table parsing failed: {str(e)}")
            return {}

    def _normalize_sector_name(self, sector: str) -> str:
        """Standardize sector names with tech aliases"""
        sector = sector.replace('&', 'and').strip().lower()
        if any(x in sector for x in ['tech', 'information', 'software', 'semiconductor']):
            return 'Information Technology'
        if 'health' in sector:
            return 'Health Care'
        return sector.title()

    def _is_valid_ticker(self, ticker: str) -> bool:
        """Validate ticker format with tech exceptions"""
        return (3 <= len(ticker) <= 5 and 
                '.' not in ticker and 
                not ticker.startswith('BRK'))

    def get_historical_data(self, tickers: list, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        """Fetch historical data with enhanced error handling"""
        for attempt in range(3):
            try:
                # Download market data
                data = yf.download(
                    tickers=tickers,
                    period=period,
                    interval=interval,
                    group_by='ticker',
                    progress=False,
                    threads=True,
                    timeout=15
                )
                
                if data.empty:
                    continue
                    
                # Process and clean data
                df = data.stack(level=0, future_stack=True).reset_index()
                df = df.rename(columns={'level_1': 'Ticker'})
                
                # Add technical indicators
                return FeatureEngineer().add_technical_indicators(df)
                
            except Exception as e:
                self.logger.warning(f"Download attempt {attempt+1} failed: {str(e)}")
                time.sleep(2)
                
        self.logger.error("All download attempts failed")
        return pd.DataFrame()

    def get_sector_stocks(self, sector: str, top_n: int = 5) -> List[str]:
        """Get top performing stocks in a sector with fallback to predefined list"""
        try:
            sector = self._normalize_sector_name(sector)
            tickers = self.sector_map.get(sector, [])
            
            if not tickers:
                return []

            # Try to get performance data, but fallback to predefined list if it fails
            try:
                data = yf.download(tickers, period="1mo", progress=False)['Close']
                returns = data.pct_change(fill_method=None).mean().sort_values(ascending=False)
                result = returns.dropna().head(top_n).index.tolist()
                if result:  # If we got results, return them
                    return result
            except Exception as e:
                self.logger.warning(f"Performance calculation failed, using predefined list: {str(e)}")
            
            # Fallback: return first N stocks from predefined list
            return tickers[:top_n]
            
        except Exception as e:
            self.logger.error(f"Sector stock fetch failed: {str(e)}")
            return []

    def get_realtime_quotes(self, tickers: list) -> Dict[str, Dict]:
        """Get real-time quotes with failover handling"""
        quotes = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                quotes[ticker] = {
                    'price': info.get('currentPrice'),
                    'volume': info.get('regularMarketVolume'),
                    'pe_ratio': info.get('trailingPE'),
                    'market_cap': info.get('marketCap'),
                    'sector': info.get('sector', 'Unknown')
                }
            except Exception as e:
                self.logger.warning(f"Failed to get real-time data for {ticker}: {str(e)}")
                quotes[ticker] = {}
        return quotes
    
    def get_sector_performance(self) -> pd.DataFrame:
        """Get sector performance data from FMP API"""
        try:
            url = f"https://financialmodelingprep.com/api/v3/sectors-performance?apikey={'wUrCxc5o79x1XnRLslwlpia46AIxr52U'}"
            response = requests.get(url)
            return pd.DataFrame(response.json())
        except Exception as e:
            self.logger.error(f"Sector performance fetch failed: {str(e)}")
            return pd.DataFrame()

