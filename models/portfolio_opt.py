import numpy as np
import pandas as pd
from scipy.optimize import minimize
import tensorflow as tf
from models.rl_agent import DQNAgent

class HybridPortfolioOptimizer:
    """Combines Modern Portfolio Theory with RL recommendations"""
    
    def __init__(self, risk_profile: str = 'Medium', model_path: str = 'models/rl_portfolio.h5'):
        self.risk_mapping = {'Low': 0.3, 'Medium': 0.5, 'High': 1.0}
        self.risk_profile = risk_profile
        self.rl_agent = self._load_rl_agent(model_path)
        
    def _load_rl_agent(self, model_path: str) -> DQNAgent:
        """Load pre-trained RL model"""
        agent = DQNAgent(input_size=4, output_size=3)  # Changed to 4 inputs for older model
        agent.model = tf.keras.models.load_model(model_path)
        return agent

    def optimize(self, returns: pd.DataFrame, state: np.ndarray) -> np.ndarray:
        """Hybrid optimization combining MPT and RL"""
        # Data validation
        if returns.empty:
            raise ValueError("Cannot optimize with empty returns data")
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("Returns must be a pandas DataFrame")
            
        # Traditional MPT optimization
        mpt_weights = self._mpt_optimization(returns)
        
        # RL-based adjustment
        rl_action = self.rl_agent.model.predict(state.reshape(1, -1), verbose=0)[0]
        return self._blend_strategies(mpt_weights, rl_action)

    def _mpt_optimization(self, returns: pd.DataFrame) -> np.ndarray:
        """Modern Portfolio Theory core implementation"""
        n_assets = len(returns.columns)
        initial_weights = np.array([1/n_assets]*n_assets)
        
        def neg_sharpe(weights):
            port_return = np.dot(weights, returns.mean())
            port_vol = np.sqrt(weights.T @ returns.cov() @ weights)
            return -port_return / port_vol
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, self.risk_mapping[self.risk_profile]) for _ in range(n_assets)]
        
        result = minimize(neg_sharpe, initial_weights,
                        method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def _blend_strategies(self, mpt_weights: np.ndarray, rl_action: np.ndarray) -> np.ndarray:
        """Combine MPT and RL recommendations"""
        adjustment = rl_action[2] - rl_action[0]  # increase - reduce
        adjusted = mpt_weights * (1 + adjustment)
        return adjusted / np.sum(adjusted)  # Re-normalize

    def get_rl_state(self, returns: pd.DataFrame, sentiment: float) -> np.ndarray:
        """Create state vector for RL model"""
        return np.array([
            returns.mean().mean(),                   # Market return
            self.risk_mapping[self.risk_profile],    # Risk level
            len(returns),                            # Time horizon/scope
            sentiment                                # Combined sentiment score
        ])
