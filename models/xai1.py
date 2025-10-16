# xai.py
import shap
from lime.lime_text import LimeTextExplainer
from lime.explanation import Explanation
import numpy as np

class XAIExplainer:
    def __init__(self):
        self.text_explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

    def explain_prediction(self, model, data, background=None):
        """Handle both 1D and 2D input shapes for SHAP"""
        # Ensure data is 2D array
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        # Create proper background distribution
        if background is None:
            background = shap.maskers.Independent(data, max_samples=100)
            
        # Create model wrapper for proper output dimensions
        def model_wrapper(x):
            return model.predict(x, verbose=0)
            
        explainer = shap.Explainer(model_wrapper, background)
        return explainer(data)

    def explain_text(self, text, predict_fn) -> Explanation:
        exp = self.text_explainer.explain_instance(
            text,
            predict_fn,
            num_features=10
        )
        return exp
