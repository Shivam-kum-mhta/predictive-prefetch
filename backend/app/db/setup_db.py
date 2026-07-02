#!/usr/bin/env python3
"""
Database setup script for the predictive prefetch system.
This script creates SQLite database tables and populates them with article data from the dataset.
"""

import sqlite3
import pandas as pd
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse

def create_database_connection(db_path: str = "articles.db") -> sqlite3.Connection:
    """Create a connection to the SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn

def create_tables(conn: sqlite3.Connection) -> None:
    """Create the articles table in the database."""
    
    create_articles_table = """
    CREATE TABLE IF NOT EXISTS articles (
        article_id TEXT PRIMARY KEY,
        category TEXT NOT NULL,
        subcategory TEXT NOT NULL,
        title TEXT NOT NULL,
        abstract TEXT,
        url TEXT NOT NULL,
        entities TEXT,  -- JSON string of entities
        keywords TEXT,  -- JSON string of keywords
        views INTEGER DEFAULT 0,  -- Number of views
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    create_indexes = [
        "CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);",
        "CREATE INDEX IF NOT EXISTS idx_articles_subcategory ON articles(subcategory);",
        "CREATE INDEX IF NOT EXISTS idx_articles_title ON articles(title);",
        "CREATE INDEX IF NOT EXISTS idx_articles_views ON articles(views);"
    ]
    
    conn.execute(create_articles_table)
    
    for index_sql in create_indexes:
        conn.execute(index_sql)
    
    conn.commit()
    print("Database tables created successfully.")

def parse_tsv_line(line: str) -> Dict[str, Any]:
    """Parse a single line from the news TSV file."""
    parts = line.strip().split('\t')
    
    if len(parts) < 6:
        return None
    
    article_id = parts[0]
    category = parts[1]
    subcategory = parts[2]
    title = parts[3]
    abstract = parts[4] if len(parts) > 4 and parts[4] else ""
    url = parts[5] if len(parts) > 5 else ""
    
    # Parse entities (7th column if exists)
    entities = "[]"
    if len(parts) > 6 and parts[6]:
        try:
            entities = parts[6]
            # Validate JSON
            json.loads(entities)
        except json.JSONDecodeError:
            entities = "[]"
    
    # Parse keywords (8th column if exists)
    keywords = "[]"
    if len(parts) > 7 and parts[7]:
        try:
            keywords = parts[7]
            # Validate JSON
            json.loads(keywords)
        except json.JSONDecodeError:
            keywords = "[]"
    
    return {
        'article_id': article_id,
        'category': category,
        'subcategory': subcategory,
        'title': title,
        'abstract': abstract,
        'url': url,
        'entities': entities,
        'keywords': keywords
    }

def load_articles_from_tsv(tsv_path: str) -> List[Dict[str, Any]]:
    """Load articles from a TSV file."""
    articles = []
    
    print(f"Loading articles from {tsv_path}...")
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"Processed {line_num} lines...")
            
            article_data = parse_tsv_line(line)
            if article_data:
                articles.append(article_data)
    
    print(f"Loaded {len(articles)} articles from {tsv_path}")
    return articles

def insert_articles_batch(conn: sqlite3.Connection, articles: List[Dict[str, Any]]) -> None:
    """Insert a batch of articles into the database."""
    
    insert_sql = """
    INSERT OR REPLACE INTO articles 
    (article_id, category, subcategory, title, abstract, url, entities, keywords)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    # Prepare data for batch insert
    data = [
        (
            article['article_id'],
            article['category'],
            article['subcategory'],
            article['title'],
            article['abstract'],
            article['url'],
            article['entities'],
            article['keywords']
        )
        for article in articles
    ]
    
    conn.executemany(insert_sql, data)
    conn.commit()

def populate_database(db_path: str = "articles.db", dataset_dir: str = "../dataset") -> None:
    """Populate the database with article data from the dataset directory."""
    
    # Get absolute paths
    dataset_path = Path(dataset_dir).resolve()
    
    # Find TSV files in the dataset directory
    tsv_files = list(dataset_path.glob("*_news.tsv"))
    
    if not tsv_files:
        print(f"No news TSV files found in {dataset_path}")
        return
    
    print(f"Found {len(tsv_files)} news TSV files:")
    for tsv_file in tsv_files:
        print(f"  - {tsv_file.name}")
    
    # Create database connection and tables
    conn = create_database_connection(db_path)
    create_tables(conn)
    
    total_articles = 0
    
    # Process each TSV file
    for tsv_file in tsv_files:
        print(f"\nProcessing {tsv_file.name}...")
        
        articles = load_articles_from_tsv(str(tsv_file))
        
        if articles:
            # Insert in batches for better performance
            batch_size = 1000
            for i in range(0, len(articles), batch_size):
                batch = articles[i:i + batch_size]
                insert_articles_batch(conn, batch)
                print(f"Inserted batch {i//batch_size + 1}/{(len(articles)-1)//batch_size + 1}")
            
            total_articles += len(articles)
    
    # Get final statistics
    cursor = conn.execute("SELECT COUNT(*) as count FROM articles")
    final_count = cursor.fetchone()['count']
    
    cursor = conn.execute("SELECT category, COUNT(*) as count FROM articles GROUP BY category ORDER BY count DESC")
    categories = cursor.fetchall()
    
    print(f"\nDatabase populated successfully!")
    print(f"Total articles: {final_count}")
    print(f"\nArticles by category:")
    for cat in categories:
        print(f"  {cat['category']}: {cat['count']}")
    
    conn.close()

def get_database_stats(db_path: str = "articles.db") -> Dict[str, Any]:
    """Get statistics about the database."""
    conn = create_database_connection(db_path)
    
    stats = {}
    
    # Total articles
    cursor = conn.execute("SELECT COUNT(*) as count FROM articles")
    stats['total_articles'] = cursor.fetchone()['count']
    
    # Total views
    cursor = conn.execute("SELECT COALESCE(SUM(views), 0) as total_views FROM articles")
    stats['total_views'] = cursor.fetchone()['total_views']
    
    # Articles by category
    cursor = conn.execute("SELECT category, COUNT(*) as count FROM articles GROUP BY category ORDER BY count DESC")
    stats['by_category'] = [dict(row) for row in cursor.fetchall()]
    
    # Articles by subcategory
    cursor = conn.execute("SELECT subcategory, COUNT(*) as count FROM articles GROUP BY subcategory ORDER BY count DESC LIMIT 10")
    stats['by_subcategory'] = [dict(row) for row in cursor.fetchall()]
    
    # Most viewed articles
    cursor = conn.execute("SELECT article_id, title, category, views FROM articles WHERE views > 0 ORDER BY views DESC LIMIT 5")
    stats['most_viewed'] = [dict(row) for row in cursor.fetchall()]
    
    # Sample articles
    cursor = conn.execute("SELECT article_id, title, category, views FROM articles ORDER BY RANDOM() LIMIT 5")
    stats['sample_articles'] = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return stats

def increment_article_views(db_path: str, article_ids: List[str]) -> None:
    """
    Increment the view count for a list of articles.
    
    Args:
        db_path: Path to the database file
        article_ids: List of article IDs to increment views for
    """
    if not article_ids:
        return
    
    conn = create_database_connection(db_path)
    
    # Increment views for each article
    update_sql = "UPDATE articles SET views = views + 1 WHERE article_id = ?"
    
    for article_id in article_ids:
        conn.execute(update_sql, (article_id,))
    
    conn.commit()
    conn.close()
    print(f"Incremented views for {len(article_ids)} articles")

def reset_article_views(db_path: str = "articles.db") -> None:
    """
    Reset all article views to 0.
    
    Args:
        db_path: Path to the database file
    """
    conn = create_database_connection(db_path)
    conn.execute("UPDATE articles SET views = 0")
    conn.commit()
    conn.close()
    print("Reset all article views to 0")

def main():
    """Main function to set up the database."""
    parser = argparse.ArgumentParser(description="Set up the articles database")
    parser.add_argument("--db-path", default="articles.db", help="Path to the SQLite database file")
    parser.add_argument("--dataset-dir", default="../dataset", help="Path to the dataset directory")
    parser.add_argument("--stats-only", action="store_true", help="Only show database statistics")
    
    args = parser.parse_args()
    
    if args.stats_only:
        if os.path.exists(args.db_path):
            stats = get_database_stats(args.db_path)
            print("Database Statistics:")
            print(f"Total articles: {stats['total_articles']}")
            print(f"Total views: {stats['total_views']}")
            print("\nArticles by category:")
            for cat in stats['by_category']:
                print(f"  {cat['category']}: {cat['count']}")
            print("\nTop subcategories:")
            for subcat in stats['by_subcategory']:
                print(f"  {subcat['subcategory']}: {subcat['count']}")
            if stats.get('most_viewed'):
                print("\nMost viewed articles:")
                for article in stats['most_viewed']:
                    print(f"  {article['article_id']}: {article['title'][:60]}... (views: {article['views']})")
            print("\nSample articles:")
            for article in stats['sample_articles']:
                print(f"  {article['article_id']}: {article['title'][:60]}... (views: {article['views']})")
        else:
            print(f"Database {args.db_path} does not exist.")
    else:
        populate_database(args.db_path, args.dataset_dir)

if __name__ == "__main__":
    main()
