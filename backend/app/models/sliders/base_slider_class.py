from typing import List, Dict, Any

class Slider:
    def __init__(self, name,slider_ref, history: List[Dict[str, Any]] = None):
        self.name = name
        self.slider_ref = slider_ref
        self.history = history if history is not None else []

    def apply(self, recommendations: List[Dict[str, Any]]):
        """Apply the slider to the recommendations"""
        return recommendations
    