"""Test script for prediction pipeline - Hit Rate @ K and MRR"""

import pandas as pd
import numpy as np
import sys
import os
import time
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from app.models.prediction import predict

def load_data():
    dataset_path = os.path.join(project_root, 'dataset', 'behaviors_test_clean.tsv')
    news_path = os.path.join(project_root, 'dataset', 'news_all_clean.tsv')
    
    behaviours_df = pd.read_csv(dataset_path, sep='\t',
                               names=['Serial', 'UserID', 'Time', 'History', 'Impressions'])
    news_df = pd.read_csv(news_path, sep='\t',
                         names=['NewsID', 'Category', 'Subcategory', 'Title', 'Abstract', 
                                'URL', 'TitleEntities', 'AbstractEntities'])
    
    return behaviours_df, news_df

def prepare_test_cases(behaviours_df, max_cases=None):
    test_cases = []
    
    for idx, row in behaviours_df.iterrows():
        if max_cases and len(test_cases) >= max_cases:
            break
            
        history = row['History']
        if pd.isna(history) or not isinstance(history, str) or history.strip() == '':
            continue
        
        news_ids = history.split()
        if len(news_ids) >= 3:
            test_cases.append((news_ids[:-1], news_ids[-1]))
    
    return test_cases

def evaluate(test_cases, top_k_values=[1, 2, 4, 8, 10, 16, 32]):
    max_k = max(top_k_values)
    hit_counts = {k: 0 for k in top_k_values}
    reciprocal_ranks = []
    evaluated = 0
    
    print(f"\nEvaluating {len(test_cases)} test cases...")
    start_time = time.time()
    
    for idx, (history, target) in enumerate(test_cases):
        if idx % 100 == 0 and idx > 0:
            print(f"  {idx}/{len(test_cases)} ({idx/len(test_cases)*100:.1f}%)")
        
        try:
            result = predict({"user_history": history, "top_k": max_k})
            
            if result.get("status") != "success":
                continue
            
            predictions = result.get("predictions", [])
            if not predictions:
                continue
            
            evaluated += 1
            predicted_ids = [p["article_id"] for p in predictions]
            
            for k in top_k_values:
                if target in predicted_ids[:k]:
                    hit_counts[k] += 1
            
            if target in predicted_ids:
                reciprocal_ranks.append(1.0 / (predicted_ids.index(target) + 1))
            else:
                reciprocal_ranks.append(0.0)
                
        except:
            continue
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Evaluated: {evaluated}/{len(test_cases)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(test_cases)*1000:.1f}ms per case)")
    
    results = {'evaluated': evaluated, 'total': len(test_cases), 'hit_rate': {}}
    
    print(f"\nHit Rate @ K:")
    for k in sorted(top_k_values):
        hit_rate = hit_counts[k] / evaluated * 100 if evaluated > 0 else 0
        results['hit_rate'][k] = hit_rate
        print(f"  @ {k:2d}: {hit_rate:7.3f}%")
    
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    results['mrr'] = mrr
    print(f"\nMRR: {mrr:.6f}")
    
    return results

def main():
    print("Loading data...")
    behaviours_df, news_df = load_data()
    
    print("Preparing test cases...")
    test_cases = prepare_test_cases(behaviours_df, max_cases=10000)
    print(f"Test cases: {len(test_cases)}")
    
    results = evaluate(test_cases)
    
    results_path = os.path.join(project_root, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main()
