from .base_slider_class import Slider
from typing import List, Dict, Any
from datetime import datetime, timedelta
import numpy as np

class PopularityBias(Slider):
    def __init__(self, name,slider_ref, history: List[Dict[str, Any]] = None):
        super().__init__(name,slider_ref)

    def apply(self, recommendations: List[Dict[str, float]]):
        """Apply the slider to the recommendations"""
        
        if not recommendations:
            return recommendations

        wt = self.slider_ref["weight"]
        distribution = self.slider_ref["distribution"]
        exponential_base = self.slider_ref.get("exponential_base", 2.0)
        logarithmic_base = self.slider_ref.get("logarithmic_base", 2.0)

        # Check if views field exists in recommendations
        views = [rec.get("views", 0) for rec in recommendations]
        max_views = max(views) if views else 1
        
        # Avoid division by zero
        if max_views == 0:
            max_views = 1
        
        views_normalized = [view / max_views for view in views]

        applied_count = 0
        for idx, rec in enumerate(recommendations):
            try:
                if distribution == "linear":
                    # Linear scaling based on normalized views
                    rec["probability"] *= (1 + wt * views_normalized[idx])
                elif distribution == "exponential":
                    # Exponential scaling
                    rec["probability"] *= np.power(exponential_base, wt * views_normalized[idx])
                elif distribution == "logarithmic":
                    # Logarithmic scaling (add small epsilon to avoid log(0))
                    normalized_view = max(views_normalized[idx], 0.01)
                    rec["probability"] *= (1 + wt * np.log(normalized_view) / np.log(logarithmic_base))
                
                applied_count += 1
            except Exception as e:
                print(f"Warning: Error applying popularity bias to recommendation {idx}: {e}")
                continue

        # print(f"Popularity bias applied to {applied_count}/{len(recommendations)} recommendations")
        return recommendations
    