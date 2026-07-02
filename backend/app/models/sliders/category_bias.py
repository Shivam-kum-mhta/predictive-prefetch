from .base_slider_class import Slider
from typing import List, Dict, Any

class CategoryBias(Slider):
    def __init__(self, name, slider_ref, history: List[Dict[str, Any]] = None):
        super().__init__(name, slider_ref, history)

    def apply(self, recommendations: List[Dict[str, float]]):
        """
        Apply category bias to recommendations.
        
        Boosts articles that are in the same category as the user's 
        most recently read article.
        
        Requirements:
        1. Recommendations must be enriched with 'category' field
        2. History must be enriched with 'category' field
        """
        
        if not recommendations:
            return recommendations

        if not self.history or len(self.history) == 0:
            return recommendations

        wt = self.slider_ref.get("weight", 1.0)

        last_history_item = self.history[-1]
        
        if isinstance(last_history_item, str):
            # print("Warning: History contains article IDs only, need full article data for category bias")
            return recommendations
        
        if not isinstance(last_history_item, dict) or "category" not in last_history_item:
            # print("Warning: Last history item doesn't have category field")
            return recommendations
        
        history_category = last_history_item["category"]
        
        if not history_category:
            return recommendations

        has_categories = any(rec.get("category") for rec in recommendations)
        if not has_categories:
            # print("Warning: Recommendations don't have 'category' field, skipping category bias")
            return recommendations

        applied_count = 0
        for rec in recommendations:
            rec_category = rec.get("category")
            
            if rec_category and rec_category == history_category:
                rec["probability"] *= (1 + wt)
                applied_count += 1
        
        # print(f"Category bias applied to {applied_count}/{len(recommendations)} recommendations (category: {history_category})")
        return recommendations

