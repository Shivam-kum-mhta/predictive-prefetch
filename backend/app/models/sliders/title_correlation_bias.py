from .base_slider_class import Slider
from typing import List, Dict, Any
from datetime import datetime, timedelta
from ...embeddings.embeddings import NewsEmbeddings

try:
    import nltk
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: nltk not available, TitleCorrelationBias will not work")

class TitleCorrelationBias(Slider):
    def __init__(self, name, slider_ref, history: List[str] = None):
        """
        Initialize TitleCorrelationBias slider.
        
        Note: history can be either:
        - List of article IDs (strings) - slider will skip if not enriched
        - List of enriched article dicts with 'title' field - slider will work properly
        
        This slider also requires enriched recommendations with 'title' field.
        """
        super().__init__(name, slider_ref, history)


    def apply(self, recommendations: List[Dict[str, float]]):
        """
        Apply title correlation bias to recommendations.
        return self._apply(recommendations)
        """

        news_embeddings = NewsEmbeddings()
        wt = self.slider_ref.get("weight", 1.0)
        last_history_item = self.history[-1]
        history_id = last_history_item["article_id"]
        # print("Applying similarity")
        try:
            for rec in recommendations:
                    
                    rec_id = rec["article_id"]
                    similarity = news_embeddings.SimilarityBetweenTitles(history_id, rec_id)
                    
                    if similarity > 0:
                        # Boost probability based on title overlap
                        rec["probability"] *= (1 + wt * similarity)
                    # Don't zero out non-matching articles, just don't boost them
        except Exception as e:
            print(f"Error applying title correlation bias: {e}")
        
        return recommendations


    ## Old apply method using simple stemming and matching
    def _apply(self, recommendations: List[Dict[str, float]]):
        """
        Apply title correlation bias to recommendations.
        
        This boosts articles whose titles have words in common with 
        the most recent article in the user's history.
        
        Requirements:
        1. Recommendations must be enriched with 'title' field
        2. History must be enriched with article data (dicts with 'title' field)
        3. nltk library must be installed
        
        Uses Porter Stemmer to normalize words before comparison.
        """
        
        if not recommendations:
            return recommendations
        
        if not NLTK_AVAILABLE:
            print("Warning: nltk not available, skipping title correlation bias")
            return recommendations
        
        if not self.history or len(self.history) == 0:
            print("Warning: No history available, skipping title correlation bias")
            return recommendations

        wt = self.slider_ref.get("weight", 1.0)

        # Check if recommendations have title field
        has_titles = any(rec.get("title") for rec in recommendations)
        if not has_titles:
            print("Warning: Recommendations don't have 'title' field, skipping title correlation bias")
            print("         Make sure to use enrich=true when getting predictions")
            return recommendations

        try:
            ps = PorterStemmer()
            
            # Get the most recent history item
            last_history_item = self.history[-1]
            
            # Handle both enriched (dict) and non-enriched (string) history
            if isinstance(last_history_item, str):
                print("Warning: History contains article IDs only, need full article data for title correlation")
                return recommendations
            
            # If it's a dict, check for title
            if not isinstance(last_history_item, dict) or "title" not in last_history_item:
                print("Warning: Last history item doesn't have title field")
                return recommendations
            
            # Extract vocabulary from last viewed article's title
            history_title = last_history_item["title"]
            top_of_history_vocab = set([ps.stem(word.lower()) for word in history_title.split()])
            top_of_history_vocab_size = len(top_of_history_vocab)
            
            if top_of_history_vocab_size == 0:
                return recommendations
            
            applied_count = 0
            for rec in recommendations:
                rec_title = rec.get("title", "")
                if not rec_title:
                    continue
                
                # Extract vocabulary from recommendation title
                rec_vocab = set([ps.stem(word.lower()) for word in rec_title.split()])
                
                # Calculate overlap
                intersection_amt = len(rec_vocab & top_of_history_vocab)
                
                if intersection_amt > 0:
                    # Boost probability based on title overlap
                    correlation_score = intersection_amt / top_of_history_vocab_size
                    rec["probability"] *= (1 + wt * correlation_score)
                    applied_count += 1
                # Don't zero out non-matching articles, just don't boost them
            
            # print(f"Title correlation bias applied to {applied_count}/{len(recommendations)} recommendations")
            
        except Exception as e:
            print(f"Error applying title correlation bias: {e}")
        
        return recommendations
    