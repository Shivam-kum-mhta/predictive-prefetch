"""
Grid Search CV for Slider Parameter Optimization
-------------------------------------------------
Optimizes weights for recency_bias, category_bias, subcategory_bias
and affect_until for recency_bias.
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import List, Tuple, Dict
from collections import defaultdict
import toml
from itertools import product

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from markov_v2 import SparseMarkovChain

# Import sliders
try:
    from app.models.sliders.sliders import reorder_recommendations
    SLIDERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Sliders not available: {e}")
    SLIDERS_AVAILABLE = False
    reorder_recommendations = None

# ==================== HELPER FUNCTIONS ====================

def enrich_articles_from_dataframe(article_list, news_df, news_dict_cache=None):
    """Enrich article data with metadata from news dataframe."""
    if news_dict_cache is None:
        news_dict = {}
        for _, row in news_df.iterrows():
            news_dict[row['NewsID']] = {
                'category': row['Category'],
                'subcategory': row['Subcategory'],
                'title': row['Title'],
                'abstract': row['Abstract'],
                'url': row['URL']
            }
    else:
        news_dict = news_dict_cache
    
    enriched = []
    for article in article_list:
        article_id = article.get('article_id') if isinstance(article, dict) else article
        enriched_article = article.copy() if isinstance(article, dict) else {'article_id': article_id}
        if article_id in news_dict:
            enriched_article.update(news_dict[article_id])
        enriched.append(enriched_article)
    
    return enriched


def load_data():
    """Load test dataset and news metadata."""
    print("Loading test data...")
    behaviours_test_df = pd.read_csv('../../../dataset/behaviors_test_clean.tsv', sep='\t',
                                     names=['Serial', 'UserID', 'Time', 'History', 'Impressions'])
    news_all_df = pd.read_csv('../../../dataset/news_all_clean.tsv', sep='\t',
                              names=['NewsID', 'Category', 'Subcategory', 'Title', 'Abstract',
                                     'URL', 'TitleEntities', 'AbstractEntities'])
    print(f"  Test behaviors: {len(behaviours_test_df):,}")
    print(f"  News articles: {len(news_all_df):,}")
    return behaviours_test_df, news_all_df


def prepare_test_cases(behaviours_df, news_df, min_history_length=1):
    """Prepare test cases from behaviors data."""
    valid_news_ids = set(news_df['NewsID'].unique())
    test_cases = []
    
    for history in behaviours_df['History']:
        if pd.isna(history) or not isinstance(history, str):
            continue
        news_ids = history.split()
        valid_sequence = [nid for nid in news_ids if nid in valid_news_ids]
        if len(valid_sequence) >= min_history_length + 1:
            history_seq = valid_sequence[:-1]
            target = valid_sequence[-1]
            test_cases.append((history_seq, target))
    
    return test_cases


def load_models():
    """Load both trained models."""
    mc_article = SparseMarkovChain()
    mc_article.load_model('markov_article.pkl')
    mc_category = SparseMarkovChain()
    mc_category.load_model('markov_category.pkl')
    return mc_article, mc_category


def predict_blended(mc_article, mc_category, current_article, current_category,
                    news_to_category, alpha=0.95, top_k=48):
    """Generate blended predictions from both models."""
    article_preds = {}
    if current_article in mc_article.state_to_idx:
        article_preds_list = mc_article.predict_next(current_article, top_k=top_k)
        article_preds = dict(article_preds_list)
    
    category_probs = {}
    if current_category in mc_category.state_to_idx:
        category_preds_list = mc_category.predict_next(current_category, top_k=30)
        category_probs = dict(category_preds_list)
    
    blended_scores = {}
    for article_id, article_prob in article_preds.items():
        article_cat = news_to_category.get(article_id)
        category_prob = category_probs.get(article_cat, 0.0)
        blended_score = alpha * article_prob + (1 - alpha) * category_prob
        blended_scores[article_id] = blended_score
    
    sorted_predictions = sorted(blended_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_predictions


# ==================== EVALUATION WITH CUSTOM SLIDER CONFIG ====================

def evaluate_with_slider_config(mc_article, mc_category, test_cases, news_df, 
                                 slider_config, alpha=0.95, top_k_values=[2, 4, 8]):
    """
    Evaluate model with custom slider configuration.
    
    Args:
        slider_config: Dict with slider parameters to override
    """
    news_to_category = dict(zip(news_df['NewsID'], news_df['Category']))
    
    # Build news cache
    news_dict_cache = {}
    for _, row in news_df.iterrows():
        news_dict_cache[row['NewsID']] = {
            'category': row['Category'],
            'subcategory': row['Subcategory'],
            'title': row['Title'],
            'abstract': row['Abstract'],
            'url': row['URL']
        }
    
    max_k = 48
    hit_counts = {k: 0 for k in top_k_values}
    reciprocal_ranks = []
    coverage_count = 0
    skipped_count = 0
    
    # Load base settings and override with slider_config
    settings_path = os.path.join(os.path.dirname(__file__), '../../../MODEL_SETTINGS.toml')
    with open(settings_path, "r") as f:
        settings = toml.load(f)
    
    # Override slider parameters
    for slider_name, params in slider_config.items():
        if slider_name in settings['sliders']:
            settings['sliders'][slider_name].update(params)
    
    # Temporarily monkey-patch the settings loader
    import app.models.sliders.sliders as sliders_module
    original_get_config = sliders_module.get_sliders_config
    sliders_module.get_sliders_config = lambda: settings['sliders']
    
    try:
        for idx, (history, target) in enumerate(test_cases):
            current_article = history[-1]
            current_category = news_to_category.get(current_article)
            
            if current_article not in mc_article.state_to_idx:
                skipped_count += 1
                continue
            if target not in mc_article.state_to_idx:
                skipped_count += 1
                continue
            if not current_category or current_category not in mc_category.state_to_idx:
                skipped_count += 1
                continue
            
            coverage_count += 1
            
            try:
                predictions = predict_blended(
                    mc_article, mc_category, current_article, current_category,
                    news_to_category, alpha=alpha, top_k=max_k
                )
                
                # Apply sliders
                formatted_predictions = [
                    {'article_id': article_id, 'probability': score}
                    for article_id, score in predictions[:max_k]
                ]
                formatted_predictions = enrich_articles_from_dataframe(formatted_predictions, news_df, news_dict_cache)
                enriched_history = [{'article_id': aid} for aid in history]
                enriched_history = enrich_articles_from_dataframe(enriched_history, news_df, news_dict_cache)
                
                formatted_predictions = reorder_recommendations(formatted_predictions, enriched_history)
                formatted_predictions.sort(key=lambda x: x.get('probability', 0), reverse=True)
                
                predicted_articles = [pred['article_id'] for pred in formatted_predictions[:max_k]]
                
                for k in top_k_values:
                    if target in predicted_articles[:k]:
                        hit_counts[k] += 1
                
                if target in predicted_articles:
                    rank = predicted_articles.index(target) + 1
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)
                    
            except Exception as e:
                skipped_count += 1
                continue
    finally:
        # Restore original function
        sliders_module.get_sliders_config = original_get_config
    
    # Calculate metrics
    results = {'hit_rate': {}}
    for k in top_k_values:
        hit_rate = hit_counts[k] / coverage_count * 100 if coverage_count > 0 else 0
        results['hit_rate'][k] = hit_rate
    
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    results['mrr'] = mrr
    
    return results


# ==================== GRID SEARCH ====================

def grid_search_sliders(mc_article, mc_category, test_cases, news_df):
    """
    Perform grid search over slider parameters.
    """
    print("\n" + "=" * 80)
    print("GRID SEARCH FOR SLIDER OPTIMIZATION")
    print("=" * 80)
    
    # Define parameter grid
    param_grid = {
        'recency_weight': [0.5, 0.8, 1.0, 1.5, 2.0, 2.5],
        'recency_affect_until': [7, 14, 30, 90, 360],
        'category_weight': [0.5, 1.0, 1.5, 2.0, 2.5],
        'subcategory_weight': [1.0, 2.0, 3.0, 4.0, 5.0]
    }
    
    print("\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    total_combinations = (len(param_grid['recency_weight']) * 
                         len(param_grid['recency_affect_until']) *
                         len(param_grid['category_weight']) *
                         len(param_grid['subcategory_weight']))
    print(f"\nTotal combinations: {total_combinations}")
    
    # Baseline (no sliders)
    baseline = {
        'hit_rate': {1: 5.63, 2: 9.17, 4: 14.64, 8: 22.21, 16: 33.37},
        'mrr': 0.1117
    }
    
    best_result = None
    best_params = None
    best_hr4 = baseline['hit_rate'][4]  # Prioritize HR@4
    
    results_log = []
    
    print("\nStarting grid search...")
    print("Optimization target: Hit Rate @ 4")
    iteration = 0
    
    for rec_w, rec_days, cat_w, subcat_w in product(
        param_grid['recency_weight'],
        param_grid['recency_affect_until'],
        param_grid['category_weight'],
        param_grid['subcategory_weight']
    ):
        iteration += 1
        
        # Create slider config
        slider_config = {
            'recency_bias': {
                'enable': True,
                'weight': rec_w,
                'affect_until': rec_days
            },
            'category_bias': {
                'enable': True,
                'weight': cat_w
            },
            'subcategory_bias': {
                'enable': True,
                'weight': subcat_w
            }
        }
        
        print(f"\n[{iteration}/{total_combinations}] Testing: rec_w={rec_w}, rec_days={rec_days}, cat_w={cat_w}, subcat_w={subcat_w}")
        
        # Evaluate
        result = evaluate_with_slider_config(
            mc_article, mc_category, test_cases, news_df,
            slider_config, alpha=0.95, top_k_values=[2, 4, 8]
        )
        
        # Log result
        result_entry = {
            'params': {
                'recency_weight': rec_w,
                'recency_affect_until': rec_days,
                'category_weight': cat_w,
                'subcategory_weight': subcat_w
            },
            'metrics': result,
            'improvement_mrr': result['mrr'] - baseline['mrr'],
            'improvement_hr2': result['hit_rate'][2] - baseline['hit_rate'][2],
            'improvement_hr4': result['hit_rate'][4] - baseline['hit_rate'][4],
            'improvement_hr8': result['hit_rate'][8] - baseline['hit_rate'][8]
        }
        results_log.append(result_entry)
        
        print(f"  MRR: {result['mrr']:.4f} (Δ{result['mrr'] - baseline['mrr']:+.4f})")
        print(f"  HR@2: {result['hit_rate'][2]:.2f}% (Δ{result['hit_rate'][2] - baseline['hit_rate'][2]:+.2f}%)")
        print(f"  HR@4: {result['hit_rate'][4]:.2f}% (Δ{result['hit_rate'][4] - baseline['hit_rate'][4]:+.2f}%)")
        print(f"  HR@8: {result['hit_rate'][8]:.2f}% (Δ{result['hit_rate'][8] - baseline['hit_rate'][8]:+.2f}%)")
        
        # Track best based on HR@4
        if result['hit_rate'][4] > best_hr4:
            best_hr4 = result['hit_rate'][4]
            best_result = result
            best_params = slider_config
            print(f"  *** NEW BEST HR@4! ***")
    
    # Print final results
    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)
    
    print("\nBASELINE (No sliders):")
    print(f"  MRR: {baseline['mrr']:.4f}")
    print(f"  HR@2: {baseline['hit_rate'][2]:.2f}%")
    print(f"  HR@4: {baseline['hit_rate'][4]:.2f}%")
    print(f"  HR@8: {baseline['hit_rate'][8]:.2f}%")
    
    if best_result:
        print("\nBEST CONFIGURATION:")
        print(f"  Recency weight: {best_params['recency_bias']['weight']}")
        print(f"  Recency affect_until: {best_params['recency_bias']['affect_until']} days")
        print(f"  Category weight: {best_params['category_bias']['weight']}")
        print(f"  Subcategory weight: {best_params['subcategory_bias']['weight']}")
        
        print("\nBEST RESULTS:")
        print(f"  MRR: {best_result['mrr']:.4f} (Δ{best_result['mrr'] - baseline['mrr']:+.4f})")
        print(f"  HR@2: {best_result['hit_rate'][2]:.2f}% (Δ{best_result['hit_rate'][2] - baseline['hit_rate'][2]:+.2f}%)")
        print(f"  HR@4: {best_result['hit_rate'][4]:.2f}% (Δ{best_result['hit_rate'][4] - baseline['hit_rate'][4]:+.2f}%)")
        print(f"  HR@8: {best_result['hit_rate'][8]:.2f}% (Δ{best_result['hit_rate'][8] - baseline['hit_rate'][8]:+.2f}%)")
    
    # Show top 5 configurations
    print("\n" + "=" * 80)
    print("TOP 5 CONFIGURATIONS BY HR@4")
    print("=" * 80)
    results_log.sort(key=lambda x: x['metrics']['hit_rate'][4], reverse=True)
    for i, entry in enumerate(results_log[:5], 1):
        p = entry['params']
        m = entry['metrics']
        print(f"\n{i}. rec_w={p['recency_weight']}, rec_days={p['recency_affect_until']}, "
              f"cat_w={p['category_weight']}, subcat_w={p['subcategory_weight']}")
        print(f"   MRR: {m['mrr']:.4f} (Δ{entry['improvement_mrr']:+.4f})")
        print(f"   HR@2: {m['hit_rate'][2]:.2f}% (Δ{entry['improvement_hr2']:+.2f}%), "
              f"HR@4: {m['hit_rate'][4]:.2f}% (Δ{entry['improvement_hr4']:+.2f}%), "
              f"HR@8: {m['hit_rate'][8]:.2f}% (Δ{entry['improvement_hr8']:+.2f}%)")


# ==================== MAIN ====================

def main():
    print("=" * 80)
    print("SLIDER PARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Load data
    behaviours_test_df, news_all_df = load_data()
    test_cases = prepare_test_cases(behaviours_test_df, news_all_df)
    print(f"Test cases: {len(test_cases):,}")
    
    # Load models
    print("\nLoading models...")
    mc_article, mc_category = load_models()
    print("Models loaded!")
    
    # Run grid search
    grid_search_sliders(mc_article, mc_category, test_cases, news_all_df)


if __name__ == "__main__":
    main()

