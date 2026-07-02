"""
Test Script for Sparse Markov Chain News Recommender
==================================================
This script tests the trained markov_model.pkl on the MINDsmall_dev_filtered_behaviors.tsv dataset.
It evaluates the model's performance using various metrics and provides detailed analysis.
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os
from typing import List, Tuple, Dict, Any
import time
from collections import defaultdict, Counter

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from markov import SparseMarkovChain

# ==================== DATA LOADING ====================

def load_test_data():
    """Load the filtered test behaviors data."""
    print("=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)
    
    # Load proper test behaviors data
    print("\nLoading behaviors_test_clean.tsv...")
    behaviours_df = pd.read_csv('../../../dataset/behaviors_test_clean.tsv', sep='\t',
                               names=['Serial', 'UserID', 'Time', 'History', 'Impressions'])
    
    print(f"  Test behaviors: {len(behaviours_df):,}")
    
    # Load all news data for context
    print("\nLoading news data for context...")
    all_news_df = pd.read_csv('../../../dataset/news_all_clean.tsv', sep='\t',
                             names=['NewsID', 'Category', 'Subcategory', 'Title', 'Abstract', 
                                    'URL', 'TitleEntities', 'AbstractEntities'])
    
    print(f"  Total news articles: {len(all_news_df):,}")
    
    return behaviours_df, all_news_df

def load_trained_model(model_path='markov_model2.pkl'):
    """Load the trained Markov model."""
    print("\n" + "=" * 80)
    print("LOADING TRAINED MODEL")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    mc = SparseMarkovChain()
    mc.load_model(model_path)
    
    print("\nModel loaded successfully!")
    mc.print_statistics()
    
    # Validate model
    print("\nValidating model...")
    validation = mc.validate_model()
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    
    return mc

# ==================== TEST DATA PREPARATION ====================

def prepare_test_cases(behaviours_df, news_df, min_history_length=2):
    """
    Prepare test cases from the filtered behaviors data.
    
    Args:
        behaviours_df: Behaviors dataframe
        news_df: News dataframe (for filtering valid NewsIDs)
        min_history_length: Minimum history length required
    
    Returns:
        List of (history, target) tuples
    """
    print(f"\nPreparing test cases (min_history={min_history_length})...")
    
    # Get valid NewsIDs
    valid_news_ids = set(news_df['NewsID'].unique())
    print(f"  Valid NewsIDs: {len(valid_news_ids):,}")
    
    test_cases = []
    skipped = 0
    
    for idx, row in behaviours_df.iterrows():
        history = row['History']
        
        if pd.isna(history) or not isinstance(history, str) or history.strip() == '':
            skipped += 1
            continue
        
        # Split space-separated NewsIDs
        news_ids = history.split()
        
        # Filter to only valid NewsIDs
        valid_sequence = [nid for nid in news_ids if nid in valid_news_ids]
        
        # Need at least min_history_length + 1 items (history + target)
        if len(valid_sequence) >= min_history_length + 1:
            # Use all but last item as history, last item as target
            history_seq = valid_sequence[:-1]
            target = valid_sequence[-1]
            test_cases.append((history_seq, target))
        else:
            skipped += 1
    
    print(f"  Test cases prepared: {len(test_cases):,}")
    print(f"  Skipped (too short or invalid): {skipped:,}")
    
    # Statistics
    if test_cases:
        lengths = [len(history) for history, _ in test_cases]
        print(f"  History length stats:")
        print(f"    Min: {min(lengths)}")
        print(f"    Max: {max(lengths)}")
        print(f"    Mean: {np.mean(lengths):.2f}")
        print(f"    Median: {np.median(lengths):.0f}")
    
    return test_cases

# ==================== MODEL EVALUATION ====================

def evaluate_model(mc, test_cases, top_k_values=[1, 2, 4, 8, 16, 32]):
    """
    Evaluate the model on test data.
    
    Metrics:
    - Hit Rate @ K: Percentage of times the target is in top K predictions
    - MRR (Mean Reciprocal Rank): Average of 1/rank of target
    - Coverage: Percentage of test cases where model can make predictions
    
    Args:
        mc: Trained SparseMarkovChain model
        test_cases: List of (history, target) tuples
        top_k_values: List of K values to evaluate
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)
    
    if not test_cases:
        print("No test cases available for evaluation!")
        return {}
    
    max_k = max(top_k_values)
    
    hit_counts = {k: 0 for k in top_k_values}
    reciprocal_ranks = []
    coverage_count = 0
    skipped_count = 0
    
    print(f"\nEvaluating {len(test_cases):,} test cases...")
    
    start_time = time.time()
    
    for idx, (history, target) in enumerate(test_cases):
        if idx % 1000 == 0 and idx > 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            print(f"  Processed {idx:,} / {len(test_cases):,} test cases... ({rate:.1f} cases/sec)")
        
        # Use the last item in history as current state
        current_state = history[-1]
        
        # Skip if state is unknown
        if current_state not in mc.state_to_idx:
            skipped_count += 1
            continue
        
        # Skip if target is unknown
        if target not in mc.state_to_idx:
            skipped_count += 1
            continue
        
        coverage_count += 1
        
        # Get predictions
        try:
            predictions = mc.predict_next(current_state, top_k=max_k)
            predicted_states = [state for state, prob in predictions]
            
            # Check if target is in top K
            for k in top_k_values:
                if target in predicted_states[:k]:
                    hit_counts[k] += 1
            
            # Calculate reciprocal rank
            if target in predicted_states:
                rank = predicted_states.index(target) + 1
                reciprocal_ranks.append(1.0 / rank)
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
    
    print(f"\nHit Rate @ K:")
    results = {'coverage': coverage, 'hit_rate': {}}
    for k in sorted(top_k_values):
        hit_rate = hit_counts[k] / coverage_count * 100 if coverage_count > 0 else 0
        results['hit_rate'][k] = hit_rate
        print(f"  Hit Rate @ {k:2d}: {hit_rate:6.2f}%")
    
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    results['mrr'] = mrr
    print(f"\nMean Reciprocal Rank (MRR): {mrr:.4f}")
    
    # Additional metrics
    if reciprocal_ranks:
        results['total_evaluated'] = len(reciprocal_ranks)
        results['avg_rank'] = np.mean([1/rr for rr in reciprocal_ranks if rr > 0])
        print(f"Average Rank (when target found): {results['avg_rank']:.2f}")
    
    print("-" * 80)
    
    return results

def analyze_predictions(mc, test_cases, news_df, num_examples=10):
    """
    Analyze prediction examples in detail.
    
    Args:
        mc: Trained model
        test_cases: Test cases
        news_df: News dataframe for looking up categories and titles
        num_examples: Number of examples to show
    """
    print("\n" + "=" * 80)
    print("PREDICTION EXAMPLES")
    print("=" * 80)
    
    # Create NewsID to Category and Title mappings
    news_to_cat = dict(zip(news_df['NewsID'], news_df['Category']))
    news_to_title = dict(zip(news_df['NewsID'], news_df['Title']))
    
    valid_examples = []
    for history, target in test_cases:
        current_state = history[-1]
        if current_state in mc.state_to_idx and target in mc.state_to_idx:
            valid_examples.append((history, target))
            if len(valid_examples) >= num_examples:
                break
    
    print(f"\nShowing {len(valid_examples)} detailed prediction examples:")
    
    for i, (history, target) in enumerate(valid_examples, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}:")
        print(f"  History length: {len(history)}")
        
        # Show last 5 items in history with categories and titles
        print(f"  Last 5 history items:")
        for news_id in history[-5:]:
            category = news_to_cat.get(news_id, 'unknown')
            title = news_to_title.get(news_id, 'Unknown Title')
            print(f"    {news_id} ({category}) - {title[:80]}{'...' if len(title) > 80 else ''}")
        
        current_state = history[-1]
        target_cat = news_to_cat.get(target, 'unknown')
        target_title = news_to_title.get(target, 'Unknown Title')
        print(f"  Target: {target} ({target_cat}) - {target_title[:80]}{'...' if len(target_title) > 80 else ''}")
        
        # Get predictions
        predictions = mc.predict_next(current_state, top_k=10)
        print(f"  Top 10 predictions:")
        for rank, (pred_state, prob) in enumerate(predictions, 1):
            pred_cat = news_to_cat.get(pred_state, 'unknown')
            pred_title = news_to_title.get(pred_state, 'Unknown Title')
            marker = " ← TARGET!" if pred_state == target else ""
            print(f"    {rank:2d}. {pred_state} ({pred_cat}): {prob:.6f} - {pred_title[:60]}{'...' if len(pred_title) > 60 else ''}{marker}")

def analyze_model_coverage(mc, test_cases, news_df):
    """Analyze model coverage and state distribution."""
    print("\n" + "=" * 80)
    print("MODEL COVERAGE ANALYSIS")
    print("=" * 80)
    
    # Analyze state coverage
    all_states_in_test = set()
    unknown_states = set()
    
    for history, target in test_cases:
        for state in history:
            all_states_in_test.add(state)
            if state not in mc.state_to_idx:
                unknown_states.add(state)
        all_states_in_test.add(target)
        if target not in mc.state_to_idx:
            unknown_states.add(target)
    
    print(f"\nState Coverage Analysis:")
    print(f"  Total unique states in test data: {len(all_states_in_test):,}")
    print(f"  States known by model: {len(all_states_in_test - unknown_states):,}")
    print(f"  Unknown states: {len(unknown_states):,}")
    print(f"  Coverage: {(len(all_states_in_test - unknown_states) / len(all_states_in_test) * 100):.2f}%")
    
    # Analyze category distribution
    news_to_cat = dict(zip(news_df['NewsID'], news_df['Category']))
    
    print(f"\nCategory Analysis:")
    category_counts = Counter()
    for state in all_states_in_test:
        if state in news_to_cat:
            category_counts[news_to_cat[state]] += 1
    
    print(f"  Top 10 categories in test data:")
    for cat, count in category_counts.most_common(10):
        print(f"    {cat}: {count:,}")

# ==================== MAIN EXECUTION ====================

def main():
    """Main test execution pipeline."""
    print("=" * 80)
    print("MARKOV MODEL TESTING")
    print("=" * 80)
    
    try:
        # Step 1: Load test data
        behaviours_df, news_df = load_test_data()
        
        # Step 2: Load trained model
        mc = load_trained_model('markov_model2.pkl')
        
        # Step 3: Prepare test cases
        print("\n" + "=" * 80)
        print("PREPARING TEST CASES")
        print("=" * 80)
        test_cases = prepare_test_cases(behaviours_df, news_df, min_history_length=2)
        
        if not test_cases:
            print("No valid test cases found!")
            return
        
        # Step 4: Evaluate model
        results = evaluate_model(mc, test_cases, top_k_values=[1, 2, 4, 8, 16, 32])
        
        # Step 5: Analyze predictions
        analyze_predictions(mc, test_cases, news_df, num_examples=10)
        
        # Step 6: Analyze model coverage
        analyze_model_coverage(mc, test_cases, news_df)
        
        # Step 7: Save results
        print("\n" + "=" * 80)
        print("SAVING TEST RESULTS")
        print("=" * 80)
        results_path = 'test_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Test results saved to {results_path}")
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
