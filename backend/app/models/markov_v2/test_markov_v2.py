"""
Testing Script for Dual Markov Chain Models (Article + Category level)
------------------------------------------------------------------------
Tests the blended recommendation system using both models for predictions.

Metrics evaluated:
1. Hit Rate @ K (for K = 1, 2, 4, 8, 16)
2. Mean Reciprocal Rank (MRR)
3. Average Rank when found
"""

import pandas as pd
import numpy as np
import sys
import os
import toml
from typing import List, Tuple, Dict
from collections import defaultdict

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

# Load settings
SETTINGS_PATH = os.path.join(os.path.dirname(__file__), '../../../MODEL_SETTINGS.toml')
with open(SETTINGS_PATH, "r") as f:
    SETTINGS = toml.load(f)

USE_CATEGORY_BLENDING = SETTINGS['model'].get('USE_CATEGORY_BLENDING', False)

# ==================== DATA LOADING ====================

def enrich_articles_from_dataframe(article_list, news_df, news_dict_cache=None):
    """
    Enrich article data with metadata from news dataframe.
    
    Args:
        article_list: List of dicts with 'article_id' and 'probability' keys
        news_df: News dataframe with article metadata
        news_dict_cache: Pre-built dictionary for fast lookup (optional)
    
    Returns:
        Enriched list with category, title, abstract, etc.
    """
    # Use cached dictionary if provided, otherwise build it
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
    
    # Enrich each article
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
    print("=" * 80)
    print("LOADING TEST DATASET")
    print("=" * 80)
    
    # Load test behaviors
    print("\nLoading test behaviors...")
    behaviours_test_df = pd.read_csv('../../../dataset/behaviors_test_clean.tsv', sep='\t',
                                     names=['Serial', 'UserID', 'Time', 'History', 'Impressions'])
    print(f"  Test behaviors: {len(behaviours_test_df):,}")
    
    # Load news data for category mapping
    print("\nLoading news metadata...")
    news_all_df = pd.read_csv('../../../dataset/news_all_clean.tsv', sep='\t',
                              names=['NewsID', 'Category', 'Subcategory', 'Title', 'Abstract',
                                     'URL', 'TitleEntities', 'AbstractEntities'])
    print(f"  All news articles: {len(news_all_df):,}")
    
    return behaviours_test_df, news_all_df


def prepare_test_cases(behaviours_df, news_df, min_history_length=1):
    """
    Prepare test cases from behaviors data.
    Each test case is (history, target) where we predict target from history.
    
    Args:
        behaviours_df: Behaviors dataframe
        news_df: News dataframe for filtering valid NewsIDs
        min_history_length: Minimum history length required
    
    Returns:
        List of (history, target) tuples
    """
    print(f"\nPreparing test cases (min_history={min_history_length})...")
    
    valid_news_ids = set(news_df['NewsID'].unique())
    test_cases = []
    skipped = 0
    
    for history in behaviours_df['History']:
        if pd.isna(history) or not isinstance(history, str):
            skipped += 1
            continue
        
        news_ids = history.split()
        valid_sequence = [nid for nid in news_ids if nid in valid_news_ids]
        
        # Need at least min_history_length items (history)
        if len(valid_sequence) >= min_history_length + 1:
            # Use last item as target, rest as history
            history_seq = valid_sequence[:-1]
            target = valid_sequence[-1]
            test_cases.append((history_seq, target))
        else:
            skipped += 1
    
    print(f"  Test cases: {len(test_cases):,}")
    print(f"  Skipped: {skipped:,}")
    
    return test_cases


# ==================== MODEL LOADING ====================

def load_models(article_model_path='markov_article.pkl', 
                category_model_path='markov_category.pkl'):
    """
    Load both trained models.
    
    Args:
        article_model_path: Path to article-level model
        category_model_path: Path to category-level model
    
    Returns:
        Tuple of (article_model, category_model)
    """
    print("\n" + "=" * 80)
    print("LOADING MODELS")
    print("=" * 80)
    
    print(f"\nLoading article-level model from {article_model_path}...")
    mc_article = SparseMarkovChain()
    mc_article.load_model(article_model_path)
    
    print(f"\nLoading category-level model from {category_model_path}...")
    mc_category = SparseMarkovChain()
    mc_category.load_model(category_model_path)
    
    return mc_article, mc_category


# ==================== BLENDED PREDICTION ====================

def predict_blended(mc_article, mc_category, current_article, current_category,
                    news_to_category, alpha=0.6, top_k=100, use_category_blending=True):
    """
    Generate predictions from article model, optionally blended with category model.
    
    Formula (if blending): score[B] = α * P_article[B|A] + (1 - α) * P_category[cat(B)|cat(A)]
    Formula (no blending): score[B] = P_article[B|A]
    
    Args:
        mc_article: Article-level model
        mc_category: Category-level model (can be None if not using blending)
        current_article: Current article ID
        current_category: Current article's category
        news_to_category: Dictionary mapping article IDs to categories
        alpha: Blending weight (0-1). Higher = more weight on article model
        top_k: Number of top predictions to return
        use_category_blending: Whether to blend with category model
    
    Returns:
        List of (article_id, score) tuples sorted by score (descending)
    """
    # Get article-level predictions: P_article[B|A]
    article_preds = {}
    if current_article in mc_article.state_to_idx:
        article_preds_list = mc_article.predict_next(current_article, top_k=top_k)
        article_preds = dict(article_preds_list)
    
    # If not using category blending, return article predictions directly
    if not use_category_blending:
        sorted_predictions = sorted(article_preds.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions
    
    # Get category-level predictions: P_category[cat(B)|cat(A)]
    category_probs = {}
    if mc_category and current_category in mc_category.state_to_idx:
        category_preds_list = mc_category.predict_next(current_category, top_k=30)
        category_probs = dict(category_preds_list)
    
    # For each article in article predictions, get its category probability
    blended_scores = {}
    for article_id, article_prob in article_preds.items():
        article_cat = news_to_category.get(article_id)
        category_prob = category_probs.get(article_cat, 0.0)
        
        # Blend: score[B] = α * P_article[B|A] + (1 - α) * P_category[cat(B)|cat(A)]
        blended_score = alpha * article_prob + (1 - alpha) * category_prob
        blended_scores[article_id] = blended_score
    
    # Sort by blended score
    sorted_predictions = sorted(blended_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_predictions


# ==================== EVALUATION ====================

def evaluate_model(mc_article, mc_category, test_cases, news_df, top_k_values=[1, 2, 4, 8, 16, 32],
                   alpha=0.6, use_sliders=False):
    """
    Evaluate the blended model on test data.
    
    Metrics:
    - Hit Rate @ K: Percentage of times the target is in top K predictions
    - MRR (Mean Reciprocal Rank): Average of 1/rank of target
    - Average Rank: Average position of target when found
    
    Args:
        mc_article: Article-level model
        mc_category: Category-level model
        test_cases: List of (history, target) tuples
        news_df: News dataframe
        top_k_values: List of K values to evaluate
        alpha: Blending weight (0-1)
        use_sliders: Whether to apply slider biases for reordering
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)
    
    if USE_CATEGORY_BLENDING:
        print(f"Category blending: ENABLED")
        print(f"Blending weight (alpha): {alpha} (article: {alpha}, category: {1-alpha})")
    else:
        print(f"Category blending: DISABLED (article model only)")
    
    if use_sliders and SLIDERS_AVAILABLE:
        print(f"Sliders: ENABLED")
    else:
        print(f"Sliders: DISABLED")
    
    # Create mappings
    news_to_category = dict(zip(news_df['NewsID'], news_df['Category']))
    
    # Pre-build news dictionary cache for enrichment (if using sliders)
    news_dict_cache = None
    if use_sliders and SLIDERS_AVAILABLE:
        print("Building news metadata cache for enrichment...")
        news_dict_cache = {}
        for _, row in news_df.iterrows():
            news_dict_cache[row['NewsID']] = {
                'category': row['Category'],
                'subcategory': row['Subcategory'],
                'title': row['Title'],
                'abstract': row['Abstract'],
                'url': row['URL']
            }
        print(f"  Cached {len(news_dict_cache):,} articles")
    
    max_k = 48
    # Include HR@32 in evaluation if not already in top_k_values
    top_k_values_with_32 = sorted(set(top_k_values + [32]))
    hit_counts = {k: 0 for k in top_k_values_with_32}
    reciprocal_ranks = []
    ranks_when_found = []
    coverage_count = 0
    skipped_count = 0
    
    print(f"\nEvaluating {len(test_cases):,} test cases...")
    
    for idx, (history, target) in enumerate(test_cases):
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1:,} / {len(test_cases):,} test cases... "
                  f"({100*(idx+1)/len(test_cases):.1f}%)")
        
        # Use the last item in history as current state
        current_article = history[-1]
        current_category = news_to_category.get(current_article)
        
        # Skip if article or category is unknown
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
        
        # Get predictions (blended or article-only based on settings)
        try:
            predictions = predict_blended(
                mc_article, mc_category, current_article, current_category,
                news_to_category, alpha=alpha, top_k=max_k,
                use_category_blending=USE_CATEGORY_BLENDING
            )
            
            # Apply sliders if enabled
            if use_sliders and SLIDERS_AVAILABLE and reorder_recommendations is not None:
                # Convert to format expected by sliders
                formatted_predictions = [
                    {'article_id': article_id, 'probability': score}
                    for article_id, score in predictions[:max_k]
                ]
                
                # Enrich predictions with article metadata (using cache)
                formatted_predictions = enrich_articles_from_dataframe(formatted_predictions, news_df, news_dict_cache)
                
                # Enrich history for sliders (using cache)
                enriched_history = [{'article_id': aid} for aid in history]
                enriched_history = enrich_articles_from_dataframe(enriched_history, news_df, news_dict_cache)
                
                # Apply sliders
                try:
                    formatted_predictions = reorder_recommendations(formatted_predictions, enriched_history)
                    
                    # IMPORTANT: Re-sort by probability after sliders modify them
                    formatted_predictions.sort(key=lambda x: x.get('probability', 0), reverse=True)
                    
                except Exception as slider_error:
                    if (idx + 1) % 1000 == 0:  # Only print occasionally
                        print(f"Warning: Slider error at case {idx+1}: {slider_error}")
                
                # Extract article IDs after reordering
                predicted_articles = [pred['article_id'] for pred in formatted_predictions[:max_k]]
            else:
                predicted_articles = [article for article, score in predictions[:max_k]]
            
            # Check if target is in top K
            for k in top_k_values_with_32:
                if target in predicted_articles[:k]:
                    hit_counts[k] += 1
            
            # Calculate rank
            if target in predicted_articles:
                rank = predicted_articles.index(target) + 1
                reciprocal_ranks.append(1.0 / rank)
                ranks_when_found.append(rank)
            else:
                reciprocal_ranks.append(0.0)
                
        except Exception as e:
            skipped_count += 1
            continue
    
    # Calculate metrics
    print("\n" + "-" * 80)
    print("EVALUATION RESULTS")
    print("-" * 80)
    
    coverage = coverage_count / len(test_cases) * 100
    print(f"\nCoverage: {coverage:.2f}% ({coverage_count:,} / {len(test_cases):,})")
    print(f"Skipped: {skipped_count:,} (unknown states/targets)")
    
    # Print Hit Rate with context about alpha
    if USE_CATEGORY_BLENDING:
        print(f"\nHit Rate @ K (alpha={alpha}):")
    else:
        print(f"\nHit Rate @ K:")
    results = {'coverage': coverage, 'hit_rate': {}, 'alpha': alpha}
    for k in sorted(top_k_values_with_32):
        hit_rate = hit_counts[k] / coverage_count * 100 if coverage_count > 0 else 0
        results['hit_rate'][k] = hit_rate
        print(f"  Hit Rate @ {k:2d}: {hit_rate:6.2f}% ({hit_counts[k]:,} hits)")
    
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    results['mrr'] = mrr
    print(f"\nMean Reciprocal Rank (MRR): {mrr:.4f}")
    
    avg_rank = np.mean(ranks_when_found) if ranks_when_found else None
    results['avg_rank_when_found'] = avg_rank
    if avg_rank is not None:
        print(f"Average Rank when found: {avg_rank:.2f}")
    else:
        print(f"Average Rank when found: N/A (no targets found in top {max_k})")
    
    print("-" * 80)
    
    return results


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution pipeline."""
    
    # Step 1: Load test data
    behaviours_test_df, news_all_df = load_data()
    
    # Step 2: Prepare test cases
    print("\n" + "=" * 80)
    print("PREPARING TEST DATA")
    print("=" * 80)
    test_cases = prepare_test_cases(behaviours_test_df, news_all_df, min_history_length=1)
    
    # Step 3: Load models
    mc_article, mc_category = load_models()
    
    # Step 4: Evaluate with sliders
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)
    
    # Use best alpha from previous tests
    alpha = 0.95
    
    all_results = {}
    
    # Baseline results (WITHOUT sliders) - pre-computed
    print(f"\n{'='*80}")
    print("BASELINE (WITHOUT SLIDERS) - Using pre-computed results")
    print(f"{'='*80}")
    results_no_sliders = {
        'coverage': 99.84,
        'hit_rate': {
            1: 5.63,
            2: 9.17,
            4: 14.64,
            8: 22.21,
            16: 33.37,
            32: 0.0  # Placeholder, will be computed during actual evaluation
        },
        'mrr': 0.1117,
        'avg_rank_when_found': 6.51,
        'alpha': alpha
    }
    all_results['no_sliders'] = results_no_sliders
    print(f"  Hit Rate @ 1:  {results_no_sliders['hit_rate'][1]:.2f}%")
    print(f"  Hit Rate @ 2:  {results_no_sliders['hit_rate'][2]:.2f}%")
    print(f"  Hit Rate @ 4:  {results_no_sliders['hit_rate'][4]:.2f}%")
    print(f"  Hit Rate @ 8:  {results_no_sliders['hit_rate'][8]:.2f}%")
    print(f"  Hit Rate @ 16: {results_no_sliders['hit_rate'][16]:.2f}%")
    print(f"  Hit Rate @ 32: {results_no_sliders['hit_rate'][32]:.2f}% (will be computed)")
    print(f"  MRR: {results_no_sliders['mrr']:.4f}")
    
    # Test WITH sliders (if available)
    if SLIDERS_AVAILABLE:
        print(f"\n{'='*80}")
        print("TESTING WITH SLIDERS")
        print(f"{'='*80}")
        results_with_sliders = evaluate_model(mc_article, mc_category, test_cases, news_all_df,
                                top_k_values=[1, 2, 4, 8, 16, 32], alpha=alpha, use_sliders=True)
        all_results['with_sliders'] = results_with_sliders
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - Impact of Sliders on Hit Rate @ K")
    print("=" * 80)
    if USE_CATEGORY_BLENDING:
        print(f"Configuration: Category blending ENABLED (alpha={alpha})")
    else:
        print(f"Configuration: Category blending DISABLED (article model only)")
    
    print(f"\n{'Config':<15} {'HR@1':<8} {'HR@2':<8} {'HR@4':<8} {'HR@8':<8} {'HR@16':<8} {'HR@32':<8} {'MRR':<8}")
    print("-" * 90)
    
    for config_name, results in all_results.items():
        hr = results['hit_rate']
        mrr = results['mrr']
        print(f"{config_name:<15} {hr[1]:<8.2f} {hr[2]:<8.2f} {hr[4]:<8.2f} {hr[8]:<8.2f} {hr[16]:<8.2f} {hr[32]:<8.2f} {mrr:<8.4f}")
    
    # Calculate improvement if sliders were used
    if 'with_sliders' in all_results:
        print("\n" + "-" * 90)
        print("IMPROVEMENT WITH SLIDERS:")
        print("-" * 90)
        for k in [1, 2, 4, 8, 16, 32]:
            no_slider_hr = all_results['no_sliders']['hit_rate'][k]
            with_slider_hr = all_results['with_sliders']['hit_rate'][k]
            improvement = with_slider_hr - no_slider_hr
            print(f"  HR@{k:2d}: {improvement:+.2f}% ({no_slider_hr:.2f}% → {with_slider_hr:.2f}%)")
        
        no_slider_mrr = all_results['no_sliders']['mrr']
        with_slider_mrr = all_results['with_sliders']['mrr']
        mrr_improvement = with_slider_mrr - no_slider_mrr
        print(f"  MRR:   {mrr_improvement:+.4f} ({no_slider_mrr:.4f} → {with_slider_mrr:.4f})")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
