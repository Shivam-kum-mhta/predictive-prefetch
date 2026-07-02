import toml
import pickle
import os
import sys
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from .sliders.sliders import reorder_recommendations

# Add modeltest directory to path to import the model classes
sys.path.append(os.path.join(os.path.dirname(__file__),  'markov_v1'))

try:
    from markov import SparseMarkovChain
except ImportError as e:
    print(f"Warning: Could not import SparseMarkovChain: {e}")
    SparseMarkovChain = None

# Import enrichment function and database from db module
try:
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from db.api import enrich_predictions_with_articles
    from db.articles import ArticleDatabase
except ImportError as e:
    print(f"Warning: Could not import db functions: {e}")
    enrich_predictions_with_articles = None
    ArticleDatabase = None

# Global model cache
_model_cache = {}

def get_model():
    """
    Load and cache the SparseMarkovChain model based on MODEL_SETTINGS.toml configuration.
    
    Returns:
        Tuple of (model_instance, settings)
    """
    # Load settings
    settings_path = os.path.join(os.path.dirname(__file__), '..', '..', 'MODEL_SETTINGS.toml')
    with open(settings_path, "r") as f:
        settings = toml.load(f)
    
    model_path = settings["model"]["MODEL_PATH"]
    
    # Check if model is already cached
    if model_path in _model_cache:
        return _model_cache[model_path], settings
    
    # Check if SparseMarkovChain is available
    if SparseMarkovChain is None:
        raise ImportError("SparseMarkovChain not available")
    
    # Resolve model path relative to project root
    full_model_path = os.path.join(os.path.dirname(__file__), model_path)
    
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Model file not found: {full_model_path}")
    
    print(f"Loading SparseMarkovChain model from {full_model_path}")
    
    # Load the model
    model = SparseMarkovChain()
    model.load_model(full_model_path)
    
    # Cache the model
    _model_cache[model_path] = model
    
    return model, settings

def predict_next_articles(user_history: List[str], top_k: Optional[int] = None) -> List[Tuple[str, float]]:
    """
    Predict the next articles a user might read based on their reading history.
    Uses weighted ensemble from recent history if configured.
    
    Args:
        user_history: List of article IDs that the user has read (most recent last)
        top_k: Number of top predictions to return (defaults to OUTPUT_ATMOST from settings)
    
    Returns:
        List of (article_id, probability) tuples sorted by probability (descending)
    """
    model, settings = get_model()
    
    # Get configuration from settings
    input_atmost = settings["recommendation"]["INPUT_ATMOST"]
    output_atmost = settings["recommendation"]["OUTPUT_ATMOST"]
    ensemble_config = settings["recommendation"].get("ensemble", {})
    
    # Use provided top_k or default from settings
    if top_k is None:
        top_k = output_atmost
    
    # Truncate history to what's required
    if len(user_history) > input_atmost:
        user_history = user_history[-input_atmost:]
    
    if not user_history:
        return []
    
    # Get ensemble configuration
    history_atmost = ensemble_config.get("history_atmost", 1)
    decay_factor = ensemble_config.get("exponential_decay_factor", 0.6)
    
    # If history_atmost is 1, use single-article prediction (original behavior)
    if history_atmost == 1:
        return _predict_from_single_article(model, user_history[-1], top_k, settings)
    
    # Weighted ensemble prediction from multiple articles
    return _predict_with_weighted_ensemble(
        model, 
        user_history, 
        history_atmost, 
        decay_factor, 
        top_k,
        settings
    )


def _predict_from_single_article(model, article_id: str, top_k: int, settings: Dict) -> List[Tuple[str, float]]:
    """
    Predict from a single article (original behavior).
    
    Args:
        model: The Markov model
        article_id: The article to predict from
        top_k: Number of predictions to return
        settings: Model settings
    
    Returns:
        List of (article_id, probability) tuples
    """
    # Check if article is known to the model
    if hasattr(model, 'state_to_idx') and article_id not in model.state_to_idx:
        # Fallback to most frequent states
        if hasattr(model, 'state_frequencies') and model.state_frequencies is not None:
            top_indices = np.argsort(model.state_frequencies)[-top_k:][::-1]
            total_freq = np.sum(model.state_frequencies)
            return [(model.idx_to_state[idx], model.state_frequencies[idx] / total_freq) 
                   for idx in top_indices if model.state_frequencies[idx] > 0]
        else:
            return []
    
    # Get predictions from the model
    predictions = model.predict_next(article_id, top_k=top_k)
    return predictions


def _predict_with_weighted_ensemble(
    model, 
    user_history: List[str], 
    history_atmost: int, 
    decay_factor: float, 
    top_k: int,
    settings: Dict
) -> List[Tuple[str, float]]:
    """
    Predict using weighted ensemble from recent history.
    
    Args:
        model: The Markov model
        user_history: List of article IDs (most recent last)
        history_atmost: How many recent articles to consider
        decay_factor: Exponential decay factor for weights
        top_k: Number of final predictions to return
        settings: Model settings
    
    Returns:
        List of (article_id, probability) tuples sorted by probability (descending)
    """
    # Get the last N articles from history
    recent_history = user_history
    
    # Calculate exponential decay weights (most recent gets highest weight)
    weights = []
    for i in range(len(recent_history)):
        # Index 0 is oldest in recent_history, -1 is most recent
        # Most recent should get weight 1.0, older articles get decayed weights
        position_from_end = len(recent_history) - 1 - i
        weight = decay_factor ** position_from_end
        weights.append(weight)
    
    # Normalize weights to sum to 1.0
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # print(f"Ensemble: Using {len(recent_history)} articles with weights: {[f'{w:.3f}' for w in normalized_weights]}")
    
    # Aggregate predictions from each article
    ensemble_predictions = {}  # article_id -> aggregated_probability
    
    # Fetch more predictions per article to ensure diversity after aggregation
    per_article_top_k = top_k * 2
    
    for i, article_id in enumerate(recent_history):
        weight = normalized_weights[i]
        
        # Get predictions from this article
        predictions = _predict_from_single_article(model, article_id, per_article_top_k, settings)
        
        if predictions:
            # Weight and aggregate
            for pred_article_id, prob in predictions:
                weighted_prob = prob * weight
                ensemble_predictions[pred_article_id] = ensemble_predictions.get(pred_article_id, 0) + weighted_prob

    
    if not ensemble_predictions:
        return []
    
    # Sort by aggregated probability and return top_k
    sorted_predictions = sorted(
        ensemble_predictions.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_k]
    
    return sorted_predictions

# Called when POST /predict
def predict(request: Dict) -> Dict:
    """
    Main prediction endpoint handler.
    
    Expected request format:
    {
        "user_history": ["article_id_1", "article_id_2", ...],
        "top_k": 5  # optional, defaults to OUTPUT_ATMOST from settings
    }
    
    Returns:
    {
        "predictions": [
            {"article_id": "N12345", "probability": 0.25},
            {"article_id": "N67890", "probability": 0.20},
            ...
        ],
        "model_type": "sparse_markov_chain",
        "status": "success"
    }
    """
    try:
        # Extract parameters from request
        user_history = request.get("user_history", [])
        
        if not isinstance(user_history, list):
            return {
                "predictions": [],
                "model_type": "unknown",
                "status": "error",
                "error": "user_history must be a list of article IDs"
            }
        
        # Get model info and settings
        _, settings = get_model()
        output_atleast = settings["recommendation"].get("OUTPUT_ATLEAST", 0)
        
        # Predict next articles
        predictions = predict_next_articles(user_history)
        
        # Format predictions
        formatted_predictions = [
            {"article_id": article_id, "probability": float(prob)}
            for article_id, prob in predictions
        ]

        # Enrich predictions with article properties BEFORE applying sliders
        formatted_predictions = enrich_predictions_with_articles(formatted_predictions)
         
        # Enrich history with article properties for sliders (especially TitleCorrelationBias)
        history_as_predictions = [
            {"article_id": article_id, "probability": 0.0}
            for article_id in user_history
        ]
        enriched_history_list = enrich_predictions_with_articles(history_as_predictions)
        enriched_history = enriched_history_list

        # Apply sliders (now with enriched data for both predictions and history)
        formatted_predictions = reorder_recommendations(formatted_predictions, enriched_history)

        # Check if we need to fill with top viewed articles from same category
        if output_atleast > 0 and len(formatted_predictions) < output_atleast:
            needed = output_atleast - len(formatted_predictions)
            # print(f"Only {len(formatted_predictions)} predictions, need {output_atleast}. Filling {needed} with top viewed from same category...")
            
            # Get category from last article in enriched history
            filler_articles = []
            if enriched_history and len(enriched_history) > 0 and ArticleDatabase is not None:
                try:
                    last_article = enriched_history[-1]
                    
                    # Extract category from enriched history
                    if isinstance(last_article, dict) and "category" in last_article:
                        category = last_article["category"]
                        
                        # Get existing article IDs to exclude
                        existing_ids = [pred["article_id"] for pred in formatted_predictions]
                        history_ids = [
                            item["article_id"] if isinstance(item, dict) else item 
                            for item in enriched_history
                        ]
                        exclude_ids = list(set(existing_ids + history_ids))
                        
                        # Fetch top viewed articles from same category
                        db = ArticleDatabase()
                        filler_articles = db.get_most_viewed_by_category(
                            category=category,
                            limit=needed,
                            exclude_ids=exclude_ids
                        )
                        
                        # Format filler articles with low probability
                        for article in filler_articles:
                            formatted_predictions.append({
                                "article_id": article["article_id"],
                                "probability": 0.001,  # Very low probability to indicate filler
                                "category": article.get("category"),
                                "subcategory": article.get("subcategory"),
                                "title": article.get("title"),
                                "abstract": article.get("abstract"),
                                "url": article.get("url"),
                                "entities": article.get("entities", []),
                                "keywords": article.get("keywords", []),
                                "views": article.get("views", 0),
                                "created_at": str(article.get("created_at")) if article.get("created_at") else None
                            })
                        
                        print(f"Added {len(filler_articles)} filler articles from category '{category}'")
                    else:
                        print("Warning: Could not extract category from history for filler articles")
                except Exception as e:
                    print(f"Warning: Failed to add filler articles: {e}")

        return {
            "predictions": formatted_predictions,
            "status": "success"
        }
            
    except Exception as e:
        return {
            "predictions": [],
            "status": "error",
            "error": str(e)
        }