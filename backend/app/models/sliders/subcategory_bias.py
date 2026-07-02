from .base_slider_class import Slider
from typing import List, Dict, Any

class SubcategoryBias(Slider):
    def __init__(self, name, slider_ref, history: List[Dict[str, Any]] = None):
        super().__init__(name, slider_ref, history)

    def apply(self, recommendations: List[Dict[str, float]]):
        """
        Apply subcategory bias to recommendations.
        
        Boosts articles that are in the same subcategory as the user's 
        most recently read article.
        
        Requirements:
        1. Recommendations must be enriched with 'subcategory' field
        2. History must be enriched with 'subcategory' field
        """
        
        if not recommendations:
            return recommendations

        if not self.history or len(self.history) == 0:
            return recommendations

        wt = self.slider_ref.get("weight", 1.0)

        last_history_item = self.history[-1]
        
        if isinstance(last_history_item, str):
            print("Warning: History contains article IDs only, need full article data for subcategory bias")
            return recommendations
        
        if not isinstance(last_history_item, dict) or "subcategory" not in last_history_item:
            print("Warning: Last history item doesn't have subcategory field")
            return recommendations
        
        history_subcategory = last_history_item["subcategory"]
        
        if not history_subcategory:
            return recommendations

        has_subcategories = any(rec.get("subcategory") for rec in recommendations)
        if not has_subcategories:
            print("Warning: Recommendations don't have 'subcategory' field, skipping subcategory bias")
            return recommendations

        applied_count = 0
        for rec in recommendations:
            rec_subcategory = rec.get("subcategory")
            
            if rec_subcategory and rec_subcategory == history_subcategory:
                rec["probability"] *= (1 + wt)
                applied_count += 1
        
        # print(f"Subcategory bias applied to {applied_count}/{len(recommendations)} recommendations (subcategory: {history_subcategory})")
        return recommendations

