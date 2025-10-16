import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class FinancialPreprocessor:
    """Preprocess financial data for ML models with traceability"""
    
    def __init__(self, numerical_features: list, categorical_features: list):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self._get_numeric_pipeline(), numerical_features),
                ('cat', self._get_categorical_pipeline(), categorical_features)
            ]
        )
    
    def _get_numeric_pipeline(self) -> Pipeline:
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    
    def _get_categorical_pipeline(self) -> Pipeline:
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw financial data"""
        return (
            df.drop_duplicates()
            .dropna(subset=['Close', 'Volume'])
            .ffill()
            .pipe(self._remove_outliers)
        )
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full preprocessing pipeline"""
        return self.preprocessor.fit_transform(df)
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        for col in self.numerical_features:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[~((df[col] < (q1 - 1.5 * iqr)) | 
                     (df[col] > (q3 + 1.5 * iqr)))]
        return df

class FeatureEngineer:
    """Create technical indicators and fundamental features"""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical analysis features"""
        df['RSI'] = FeatureEngineer._calculate_rsi(df['Close'])
        df['MACD'] = FeatureEngineer._calculate_macd(df['Close'])
        df['Bollinger_Upper'], df['Bollinger_Lower'] = \
            FeatureEngineer._calculate_bollinger_bands(df['Close'])
        return df
    
    @staticmethod
    def _calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_macd(series: pd.Series, 
                       fast: int = 12, 
                       slow: int = 26, 
                       signal: int = 9) -> pd.Series:
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd.ewm(span=signal).mean()
    
    @staticmethod
    def _calculate_bollinger_bands(series: pd.Series, 
                                  window: int = 20, 
                                  num_std: int = 2) -> tuple:
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower
