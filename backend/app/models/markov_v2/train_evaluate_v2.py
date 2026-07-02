"""
Training Script for Dual Markov Chain Models (Article + Category level)
with Semantic Similarity Smoothing
------------------------------------------------------------------------
This script trains two sparse Markov chain models:
1. Article-level model: States are individual articles
2. Category-level model: States are article categories
"""

import pandas as pd
import numpy as np
import sys
import os
import time
from typing import List, Dict

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from markov_v2 import SparseMarkovChain
from app.embeddings.embeddings import NewsEmbeddings

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


def prepare_article_sequences(behaviours_df, news_df, min_length=2):
    """
    Extract reading history sequences as article IDs.
    
    Args:
        behaviours_df: Behaviors dataframe
        news_df: News dataframe (for filtering valid NewsIDs)
        min_length: Minimum sequence length to include
    
    Returns:
        List of sequences (each sequence is a list of NewsIDs)
    """
    print(f"\nPreparing article sequences (min_length={min_length})...")
    
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
    
    print(f"  Article sequences extracted: {len(sequences):,}")
    print(f"  Skipped (too short or invalid): {skipped:,}")
    
    # Statistics
    lengths = [len(seq) for seq in sequences]
    print(f"  Sequence length stats:")
    print(f"    Min: {min(lengths)}")
    print(f"    Max: {max(lengths)}")
    print(f"    Mean: {np.mean(lengths):.2f}")
    print(f"    Median: {np.median(lengths):.0f}")
    
    return sequences


def prepare_category_sequences(article_sequences, news_df):
    """
    Convert article sequences to category sequences.
    
    Args:
        article_sequences: List of article ID sequences
        news_df: News dataframe with Category information
    
    Returns:
        List of category sequences
    """
    print("\nConverting article sequences to category sequences...")
    
    # Create article ID to category mapping
    news_to_category = dict(zip(news_df['NewsID'], news_df['Category']))
    
    category_sequences = []
    
    for article_seq in article_sequences:
        category_seq = []
        for article_id in article_seq:
            if article_id in news_to_category:
                category_seq.append(news_to_category[article_id])
        
        if len(category_seq) >= 2:
            category_sequences.append(category_seq)
    
    print(f"  Category sequences created: {len(category_sequences):,}")
    
    # Print category distribution
    from collections import Counter
    all_categories = [cat for seq in category_sequences for cat in seq]
    cat_counts = Counter(all_categories)
    print(f"\n  Top 10 categories:")
    for cat, count in cat_counts.most_common(10):
        print(f"    {cat}: {count:,}")
    
    return category_sequences


# ==================== MODEL TRAINING ====================

def train_article_model(article_sequences):
    """
    Train the article-level Markov chain model.
    
    Args:
        article_sequences: List of article ID sequences
    
    Returns:
        Trained SparseMarkovChain model
    """
    print("\n" + "=" * 80)
    print("TRAINING ARTICLE-LEVEL MARKOV CHAIN MODEL")
    print("=" * 80)
    
    start_time = time.time()
    
    # Initialize and fit model
    mc_article = SparseMarkovChain()
    mc_article.fit(article_sequences, smoothing=1e-10)
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds")
    
    # Print statistics
    print("\n")
    mc_article.print_statistics()
    
    # Validate model
    print("\nValidating model...")
    validation = mc_article.validate_model()
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    
    return mc_article


def train_category_model(category_sequences):
    """
    Train the category-level Markov chain model.
    
    Args:
        category_sequences: List of category sequences
    
    Returns:
        Trained SparseMarkovChain model
    """
    print("\n" + "=" * 80)
    print("TRAINING CATEGORY-LEVEL MARKOV CHAIN MODEL")
    print("=" * 80)
    
    start_time = time.time()
    
    # Initialize and fit model
    mc_category = SparseMarkovChain()
    mc_category.fit(category_sequences, smoothing=1e-10)
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds")
    
    # Print statistics
    print("\n")
    mc_category.print_statistics()
    
    # Validate model
    print("\nValidating model...")
    validation = mc_category.validate_model()
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    
    return mc_category


def apply_semantic_smoothing(mc_article):
    """
    Apply semantic similarity smoothing to the article-level model.
    
    Args:
        mc_article: Article-level Markov chain model
    
    Returns:
        Smoothed model (same object, modified in place)
    """
    print("\n" + "=" * 80)
    print("APPLYING SEMANTIC SIMILARITY SMOOTHING")
    print("=" * 80)
    
    # Initialize embeddings module
    embeddings = NewsEmbeddings()
    
    # Apply smoothing with specified parameters
    start_time = time.time()
    mc_article.apply_semantic_smoothing(
        embeddings_module=embeddings,
        similar_articles_n=10,
        spread_fraction=0.15
    )
    smoothing_time = time.time() - start_time
    
    print(f"\nSemantic smoothing completed in {smoothing_time:.2f} seconds")
    
    # Validate after smoothing
    print("\nValidating model after smoothing...")
    validation = mc_article.validate_model()
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    
    return mc_article


def save_models(mc_article, mc_category, output_dir='.'):
    """
    Save both models to files.
    
    Args:
        mc_article: Article-level model
        mc_category: Category-level model
        output_dir: Directory to save models
    """
    print("\n" + "=" * 80)
    print("SAVING MODELS")
    print("=" * 80)
    
    article_path = os.path.join(output_dir, 'markov_article.pkl')
    category_path = os.path.join(output_dir, 'markov_category.pkl')
    
    print(f"\nSaving article-level model to {article_path}...")
    mc_article.save_model(article_path)
    
    print(f"\nSaving category-level model to {category_path}...")
    mc_category.save_model(category_path)
    
    print("\n✓ All models saved successfully!")


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution pipeline."""
    
    # Step 1: Load data
    behaviours_train_df, news_all_df = load_data()
    
    # Step 2: Prepare article sequences
    print("\n" + "=" * 80)
    print("PREPARING TRAINING DATA")
    print("=" * 80)
    article_sequences = prepare_article_sequences(behaviours_train_df, news_all_df, min_length=2)
    
    # Step 3: Prepare category sequences
    category_sequences = prepare_category_sequences(article_sequences, news_all_df)
    
    # Step 4: Train article-level model
    mc_article = train_article_model(article_sequences)
    
    # Step 5: Train category-level model
    mc_category = train_category_model(category_sequences)
    
    # Step 6: Apply semantic smoothing to article model
    # DISABLED - Enable by setting SKIP_SMOOTHING = False
    SKIP_SMOOTHING = True
    if not SKIP_SMOOTHING:
        print("\n" + "=" * 80)
        print("NOTE: Semantic smoothing is ENABLED - this will take 1-2 hours")
        print("=" * 80)
        mc_article = apply_semantic_smoothing(mc_article)
    else:
        print("\n" + "=" * 80)
        print("NOTE: Semantic smoothing is DISABLED (SKIP_SMOOTHING = True)")
        print("To enable: Edit train_evaluate_v2.py and set SKIP_SMOOTHING = False")
        print("=" * 80)
    
    # Step 7: Save both models
    save_models(mc_article, mc_category, output_dir='.')
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nModels created:")
    print("  1. markov_article.pkl - Article-level model with semantic smoothing")
    print("  2. markov_category.pkl - Category-level model")
    print("\nNext steps: Proceed to inference and testing (Step 3)")


if __name__ == "__main__":
    main()
