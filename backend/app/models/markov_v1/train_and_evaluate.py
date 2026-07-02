"""
Training and Evaluation Script for Sparse Markov Chain News Recommender
------------------------------------------------------------------------
This script trains a sparse Markov chain model on the MIND dataset and evaluates
its performance on predicting the next news article a user will read.
"""

import pandas as pd
import numpy as np
import pickle
from markov import SparseMarkovChain
from typing import List, Tuple, Dict
import time
from collections import defaultdict

# ==================== DATA LOADING ====================

def load_data():
    """Load and prepare the MIND dataset."""
    print("=" * 80)
    print("LOADING MIND DATASET")
    print("=" * 80)
    
    # Load behaviors data - using clean datasets
    print("\nLoading behaviors data...")
    behaviours_train_df = pd.read_csv('../../../dataset/behaviors_train_clean.tsv', sep='\t', 
                                     names=['Serial', 'UserID', 'Time', 'History', 'Impressions'])
    
    print(f"  Training behaviors: {len(behaviours_train_df):,}")
    
    # Load all news data
    print("\nLoading news data...")
    news_all_df = pd.read_csv('../../../dataset/news_all_clean.tsv', sep='\t',
                              names=['NewsID', 'Category', 'Subcategory', 'Title', 'Abstract', 
                                     'URL', 'TitleEntities', 'AbstractEntities'])
    
    print(f"  All news articles: {len(news_all_df):,}")
    
    return behaviours_train_df, news_all_df


def prepare_sequences(behaviours_df, news_df, min_length=2):
    """
    Extract reading history sequences from behaviors data.
    
    Args:
        behaviours_df: Behaviors dataframe
        news_df: News dataframe (for filtering valid NewsIDs)
        min_length: Minimum sequence length to include
    
    Returns:
        List of sequences (each sequence is a list of NewsIDs)
    """
    print(f"\nPreparing sequences (min_length={min_length})...")
    
    # Get valid NewsIDs
    valid_news_ids = set(news_df['NewsID'].unique())
    print(f"  Valid NewsIDs: {len(valid_news_ids):,}")
    
    sequences = []
    skipped = 0
    
    for history in behaviours_df['History']:
        if pd.isna(history) or not isinstance(history, str):
            skipped += 1
            continue
        
        # Split space-separated NewsIDs
        news_ids = history.split()
        
        # Filter to only valid NewsIDs
        valid_sequence = [nid for nid in news_ids if nid in valid_news_ids]
        
        if len(valid_sequence) >= min_length:
            sequences.append(valid_sequence)
        else:
            skipped += 1
    
    print(f"  Sequences extracted: {len(sequences):,}")
    print(f"  Skipped (too short or invalid): {skipped:,}")
    
    # Statistics
    lengths = [len(seq) for seq in sequences]
    print(f"  Sequence length stats:")
    print(f"    Min: {min(lengths)}")
    print(f"    Max: {max(lengths)}")
    print(f"    Mean: {np.mean(lengths):.2f}")
    print(f"    Median: {np.median(lengths):.0f}")
    
    return sequences


def prepare_test_data(behaviours_df, news_df, min_history_length=2):
    """
    Prepare test data by splitting each sequence into (history, target).
    
    Args:
        behaviours_df: Behaviors dataframe
        news_df: News dataframe
        min_history_length: Minimum history length required
    
    Returns:
        List of (history, target) tuples
    """
    print(f"\nPreparing test data (min_history={min_history_length})...")
    
    valid_news_ids = set(news_df['NewsID'].unique())
    test_cases = []
    skipped = 0
    
    for history in behaviours_df['History']:
        if pd.isna(history) or not isinstance(history, str):
            skipped += 1
            continue
        
        news_ids = history.split()
        valid_sequence = [nid for nid in news_ids if nid in valid_news_ids]
        
        # Need at least min_history_length + 1 items (history + target)
        if len(valid_sequence) >= min_history_length + 1:
            # Use all but last item as history, last item as target
            history_seq = valid_sequence[:-1]
            target = valid_sequence[-1]
            test_cases.append((history_seq, target))
        else:
            skipped += 1
    
    print(f"  Test cases: {len(test_cases):,}")
    print(f"  Skipped: {skipped:,}")
    
    return test_cases


# ==================== MODEL TRAINING ====================

def train_model(train_sequences, model_path='markov_model.pkl'):
    """
    Train the sparse Markov chain model.
    
    Args:
        train_sequences: List of training sequences
        model_path: Path to save the trained model
    
    Returns:
        Trained SparseMarkovChain model
    """
    print("\n" + "=" * 80)
    print("TRAINING SPARSE MARKOV CHAIN MODEL")
    print("=" * 80)
    
    start_time = time.time()
    
    # Initialize and fit model
    mc = SparseMarkovChain()
    mc.fit(train_sequences, smoothing=1e-10)
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds")
    
    # Print statistics
    print("\n")
    mc.print_statistics()
    
    # Validate model
    print("\nValidating model...")
    validation = mc.validate_model()
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    
    # Save model
    print(f"\nSaving model to {model_path}...")
    mc.save_model(model_path)
    
    return mc


# ==================== MODEL EVALUATION ====================

def evaluate_model(mc, test_cases, top_k_values=[1, 5, 10, 20]):
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
    
    max_k = max(top_k_values)
    
    hit_counts = {k: 0 for k in top_k_values}
    reciprocal_ranks = []
    coverage_count = 0
    skipped_count = 0
    
    print(f"\nEvaluating {len(test_cases):,} test cases...")
    
    for idx, (history, target) in enumerate(test_cases):
        if idx % 10000 == 0 and idx > 0:
            print(f"  Processed {idx:,} / {len(test_cases):,} test cases...")
        
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
    
    print("-" * 80)
    
    return results


def analyze_predictions(mc, test_cases, news_df, num_examples=5):
    """
    Analyze a few prediction examples in detail.
    
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
    
    for i, (history, target) in enumerate(valid_examples, 1):
        print(f"\nExample {i}:")
        print(f"  History length: {len(history)}")
        
        # Show last 5 items in history with categories and titles
        print(f"  Last 5 history items:")
        for news_id in history[-5:]:
            category = news_to_cat.get(news_id, 'unknown')
            title = news_to_title.get(news_id, 'Unknown Title')
            print(f"    {news_id} ({category}) - {title}")
        
        current_state = history[-1]
        target_cat = news_to_cat.get(target, 'unknown')
        target_title = news_to_title.get(target, 'Unknown Title')
        print(f"  Target: {target} ({target_cat}) - {target_title}")
        
        # Get predictions
        predictions = mc.predict_next(current_state, top_k=10)
        print(f"  Top 10 predictions:")
        for rank, (pred_state, prob) in enumerate(predictions, 1):
            pred_cat = news_to_cat.get(pred_state, 'unknown')
            pred_title = news_to_title.get(pred_state, 'Unknown Title')
            marker = " ← TARGET!" if pred_state == target else ""
            print(f"    {rank:2d}. {pred_state} ({pred_cat}): {prob:.6f} - {pred_title}{marker}")


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution pipeline."""
    
    # Step 1: Load data
    behaviours_train_df, news_all_df = load_data()
    
    # Step 2: Prepare training sequences
    print("\n" + "=" * 80)
    print("PREPARING TRAINING DATA")
    print("=" * 80)
    train_sequences = prepare_sequences(behaviours_train_df, news_all_df, min_length=2)
    
    # Step 3: Train model - save as markov_model2.pkl
    mc = train_model(train_sequences, model_path='markov_model2.pkl')
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("Note: Use test_markov.py with proper test data for evaluation")


if __name__ == "__main__":
    main()

