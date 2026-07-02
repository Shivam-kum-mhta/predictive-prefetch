import csv
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import os

behaviors_file = '../dataset/MINDsmall_train_behaviors.tsv'
news_file = '../dataset/MINDsmall_train_news.tsv'

article_to_category = {}
print("Loading news data...")
with open(news_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) >= 2:
            article_id = row[0]
            category = row[1]
            article_to_category[article_id] = category

print(f"Loaded {len(article_to_category)} articles")

category_hour_counts = defaultdict(lambda: defaultdict(int))

print("Processing behaviors...")
with open(behaviors_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for idx, row in enumerate(reader):
        if len(row) >= 4:
            timestamp_str = row[2]
            history = row[3]
            
            try:
                dt = datetime.strptime(timestamp_str, '%m/%d/%Y %I:%M:%S %p')
                hour = dt.hour
                
                if history:
                    article_ids = history.split()
                    for article_id in article_ids:
                        if article_id in article_to_category:
                            category = article_to_category[article_id]
                            category_hour_counts[category][hour] += 1
            except Exception as e:
                pass
        
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1} behavior entries...")

print(f"\nTotal behavior entries processed")
print(f"Categories found: {len(category_hour_counts)}")

hours_4am_to_4am = list(range(4, 24)) + list(range(0, 4))
hour_labels = [f"{h:02d}:00" for h in hours_4am_to_4am]

os.makedirs('../res/category_plots', exist_ok=True)

for category in sorted(category_hour_counts.keys()):
    counts = [category_hour_counts[category][hour] for hour in hours_4am_to_4am]
    
    plt.figure(figsize=(14, 6))
    plt.bar(hour_labels, counts, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Hour of Day (Starting 4AM)', fontsize=12)
    plt.ylabel('Number of Articles Read', fontsize=12)
    plt.title(f'Article Reading Pattern for Category: {category}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    filename = f'../res/category_plots/{category}_hourly_distribution.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    total_reads = sum(counts)
    print(f"Saved: {filename} (Total reads: {total_reads})")

print(f"\nAll plots saved to ../res/category_plots/")

