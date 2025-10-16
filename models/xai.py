import shap
from lime.lime_text import LimeTextExplainer
from lime.explanation import Explanation
import numpy as np

class XAIExplainer:
    def __init__(self):
        self.text_explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
        
    def explain_prediction(self, model, data, background=None):
        # Always provide a masker/background for tabular data
        if background is None:
            background = np.random.randn(100, data.shape[1])  # 100 samples, n_features
        # CRITICAL: Use model.predict for multi-output models
        explainer = shap.Explainer(model.predict, masker=background)
        return explainer(data)
    
    def explain_text(self, text, predict_fn) -> Explanation:
        exp = self.text_explainer.explain_instance(
            text, 
            predict_fn, 
            num_features=10
        )
        return exp