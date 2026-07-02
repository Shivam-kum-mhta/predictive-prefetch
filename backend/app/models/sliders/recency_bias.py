from .base_slider_class import Slider
from typing import List, Dict, Any
from datetime import datetime, timedelta

class RecencyBias(Slider):
    def __init__(self, name,slider_ref, history: List[Dict[str, Any]] = None):
        super().__init__(name,slider_ref)

    def apply(self, recommendations: List[Dict[str, float]]):
        """Apply the slider to the recommendations"""
        
        if not recommendations:
            return recommendations

        wt = self.slider_ref.get("weight", 1.0)
        fake_date_str = self.slider_ref.get("cur_date_fake")
        affect_until = self.slider_ref.get("affect_until", 14)  # days ago
        
        if not fake_date_str:
            print("Warning: cur_date_fake not set, skipping recency bias")
            return recommendations
        
        try:
            fake_date = datetime.strptime(fake_date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            print(f"Warning: Invalid date format for cur_date_fake: {e}")
            return recommendations

        applied_to = 0
        skipped = 0
        
        for rec in recommendations:
            created_at = rec.get("created_at")
            
            if not created_at:
                # If no created_at field, skip this recommendation
                skipped += 1
                continue
            
            try:
                # Handle both string and datetime objects
                if isinstance(created_at, str):
                    date = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
                else:
                    date = created_at
                
                # Check if article is within the recency window
                days_old = (fake_date - date).days
                
                if days_old <= affect_until:
                    # Article is recent, boost its probability
                    # Linear decay: articles get less boost as they age
                    recency_factor = 1 - (days_old / affect_until)
                    rec["probability"] *= (1 + wt * recency_factor)
                    applied_to += 1
                # else: article is too old, no boost but don't zero it out
                
            except Exception as e:
                print(f"Warning: Error processing created_at for recommendation: {e}")
                skipped += 1
                continue
                
        # print(f"Recency bias applied to {applied_to}/{len(recommendations)} recommendations ({skipped} skipped)")
        return recommendations
    