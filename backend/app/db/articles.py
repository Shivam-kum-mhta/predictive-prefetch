#!/usr/bin/env python3
"""
Article data retrieval functions for the predictive prefetch system.
This module provides functions to query article data from the SQLite database.
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

class ArticleDatabase:
    """Database interface for article data retrieval."""
    
    def __init__(self, db_path: str = None):
        """Initialize the database connection."""
        if db_path is None:
            # Default to articles.db in the same directory as this file
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(current_dir, "articles.db")
        else:
            self.db_path = db_path
        self._ensure_database_exists()
    
    def _ensure_database_exists(self) -> None:
        """Ensure the database file exists."""
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"Database file {self.db_path} not found. Run setup_db.py first.")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def get_article_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get a single article by its ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM articles WHERE article_id = ?", 
                (article_id,)
            )
            row = cursor.fetchone()
            
            if row:
                article = dict(row)
                # Parse JSON fields
                article['entities'] = json.loads(article['entities']) if article['entities'] else []
                article['keywords'] = json.loads(article['keywords']) if article['keywords'] else []
                return article
            return None
        finally:
            conn.close()
    
    def get_articles_by_ids(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple articles by their IDs."""
        if not article_ids:
            return []
        
        conn = self._get_connection()
        try:
            placeholders = ','.join('?' * len(article_ids))
            cursor = conn.execute(
                f"SELECT * FROM articles WHERE article_id IN ({placeholders})", 
                article_ids
            )
            rows = cursor.fetchall()
            
            articles = []
            for row in rows:
                article = dict(row)
                # Parse JSON fields
                article['entities'] = json.loads(article['entities']) if article['entities'] else []
                article['keywords'] = json.loads(article['keywords']) if article['keywords'] else []
                articles.append(article)
            
            return articles
        finally:
            conn.close()
    
    def get_articles_by_category(self, category: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get articles by category."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM articles WHERE category = ? ORDER BY created_at DESC LIMIT ?",
                (category, limit)
            )
            rows = cursor.fetchall()
            
            articles = []
            for row in rows:
                article = dict(row)
                article['entities'] = json.loads(article['entities']) if article['entities'] else []
                article['keywords'] = json.loads(article['keywords']) if article['keywords'] else []
                articles.append(article)
            
            return articles
        finally:
            conn.close()
    
    def get_articles_by_subcategory(self, subcategory: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get articles by subcategory."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM articles WHERE subcategory = ? ORDER BY created_at DESC LIMIT ?",
                (subcategory, limit)
            )
            rows = cursor.fetchall()
            
            articles = []
            for row in rows:
                article = dict(row)
                article['entities'] = json.loads(article['entities']) if article['entities'] else []
                article['keywords'] = json.loads(article['keywords']) if article['keywords'] else []
                articles.append(article)
            
            return articles
        finally:
            conn.close()
    
    def search_articles(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search articles by title or abstract."""
        conn = self._get_connection()
        try:
            search_term = f"%{query}%"
            cursor = conn.execute(
                """SELECT * FROM articles 
                   WHERE title LIKE ? OR abstract LIKE ?
                   ORDER BY created_at DESC LIMIT ?""",
                (search_term, search_term, limit)
            )
            rows = cursor.fetchall()
            
            articles = []
            for row in rows:
                article = dict(row)
                article['entities'] = json.loads(article['entities']) if article['entities'] else []
                article['keywords'] = json.loads(article['keywords']) if article['keywords'] else []
                articles.append(article)
            
            return articles
        finally:
            conn.close()
    
    def get_random_articles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get random articles."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM articles ORDER BY RANDOM() LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            
            articles = []
            for row in rows:
                article = dict(row)
                article['entities'] = json.loads(article['entities']) if article['entities'] else []
                article['keywords'] = json.loads(article['keywords']) if article['keywords'] else []
                articles.append(article)
            
            return articles
        finally:
            conn.close()
    
    def get_recent_articles(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the most recent articles."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM articles ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            
            articles = []
            for row in rows:
                article = dict(row)
                article['entities'] = json.loads(article['entities']) if article['entities'] else []
                article['keywords'] = json.loads(article['keywords']) if article['keywords'] else []
                articles.append(article)
            
            return articles
        finally:
            conn.close()
    
    def get_categories(self) -> List[Dict[str, Any]]:
        """Get all categories with article counts."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT category, COUNT(*) as count FROM articles GROUP BY category ORDER BY count DESC"
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_subcategories(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all subcategories with article counts."""
        conn = self._get_connection()
        try:
            if category:
                cursor = conn.execute(
                    "SELECT subcategory, COUNT(*) as count FROM articles WHERE category = ? GROUP BY subcategory ORDER BY count DESC",
                    (category,)
                )
            else:
                cursor = conn.execute(
                    "SELECT subcategory, COUNT(*) as count FROM articles GROUP BY subcategory ORDER BY count DESC"
                )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_article_count(self) -> int:
        """Get total number of articles."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) as count FROM articles")
            return cursor.fetchone()['count']
        finally:
            conn.close()
    
    def get_article_stats(self) -> Dict[str, Any]:
        """Get comprehensive article statistics."""
        conn = self._get_connection()
        try:
            stats = {}
            
            # Total articles
            cursor = conn.execute("SELECT COUNT(*) as count FROM articles")
            stats['total_articles'] = cursor.fetchone()['count']
            
            # Total views
            cursor = conn.execute("SELECT COALESCE(SUM(views), 0) as total_views FROM articles")
            stats['total_views'] = cursor.fetchone()['total_views']
            
            # Average views per article
            cursor = conn.execute("SELECT COALESCE(AVG(views), 0) as avg_views FROM articles")
            stats['avg_views'] = round(cursor.fetchone()['avg_views'], 2)
            
            # Articles by category
            stats['by_category'] = self.get_categories()
            
            # Articles by subcategory (top 10)
            cursor = conn.execute(
                "SELECT subcategory, COUNT(*) as count FROM articles GROUP BY subcategory ORDER BY count DESC LIMIT 10"
            )
            stats['by_subcategory'] = [dict(row) for row in cursor.fetchall()]
            
            # Articles with entities
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM articles WHERE entities != '[]' AND entities IS NOT NULL"
            )
            stats['articles_with_entities'] = cursor.fetchone()['count']
            
            # Articles with keywords
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM articles WHERE keywords != '[]' AND keywords IS NOT NULL"
            )
            stats['articles_with_keywords'] = cursor.fetchone()['count']
            
            return stats
        finally:
            conn.close()
    
    def validate_article_ids(self, article_ids: List[str]) -> Dict[str, bool]:
        """Validate which article IDs exist in the database."""
        if not article_ids:
            return {}
        
        conn = self._get_connection()
        try:
            placeholders = ','.join('?' * len(article_ids))
            cursor = conn.execute(
                f"SELECT article_id FROM articles WHERE article_id IN ({placeholders})",
                article_ids
            )
            existing_ids = {row['article_id'] for row in cursor.fetchall()}
            
            return {article_id: article_id in existing_ids for article_id in article_ids}
        finally:
            conn.close()
    
    def increment_article_views(self, article_ids: List[str]) -> int:
        """
        Increment the view count for a list of articles.
        
        Args:
            article_ids: List of article IDs to increment views for
            
        Returns:
            Number of articles updated
        """
        if not article_ids:
            return 0
        
        conn = self._get_connection()
        try:
            update_sql = "UPDATE articles SET views = views + 1 WHERE article_id = ?"
            updated_count = 0
            
            for article_id in article_ids:
                cursor = conn.execute(update_sql, (article_id,))
                updated_count += cursor.rowcount
            
            conn.commit()
            return updated_count
        finally:
            conn.close()
    
    def get_most_viewed_articles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most viewed articles.
        
        Args:
            limit: Maximum number of articles to return
            
        Returns:
            List of articles ordered by views (descending)
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM articles WHERE views > 0 ORDER BY views DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            
            articles = []
            for row in rows:
                article = dict(row)
                article['entities'] = json.loads(article['entities']) if article['entities'] else []
                article['keywords'] = json.loads(article['keywords']) if article['keywords'] else []
                articles.append(article)
            
            return articles
        finally:
            conn.close()
    
    def get_most_viewed_by_category(self, category: str, limit: int = 10, exclude_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get the most viewed articles from a specific category.
        
        Args:
            category: Category to filter by
            limit: Maximum number of articles to return
            exclude_ids: List of article IDs to exclude from results
            
        Returns:
            List of articles ordered by views (descending)
        """
        conn = self._get_connection()
        try:
            if exclude_ids:
                placeholders = ','.join('?' * len(exclude_ids))
                query = f"""
                    SELECT * FROM articles 
                    WHERE category = ? AND article_id NOT IN ({placeholders})
                    ORDER BY views DESC, created_at DESC 
                    LIMIT ?
                """
                params = [category] + exclude_ids + [limit]
            else:
                query = """
                    SELECT * FROM articles 
                    WHERE category = ?
                    ORDER BY views DESC, created_at DESC 
                    LIMIT ?
                """
                params = [category, limit]
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            articles = []
            for row in rows:
                article = dict(row)
                article['entities'] = json.loads(article['entities']) if article['entities'] else []
                article['keywords'] = json.loads(article['keywords']) if article['keywords'] else []
                articles.append(article)
            
            return articles
        finally:
            conn.close()
    
    def set_article_views(self, article_id: str, views: int) -> bool:
        """
        Set the view count for a specific article.
        
        Args:
            article_id: Article ID
            views: Number of views to set
            
        Returns:
            True if the article was updated, False otherwise
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "UPDATE articles SET views = ? WHERE article_id = ?",
                (views, article_id)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def reset_all_views(self) -> int:
        """
        Reset all article views to 0.
        
        Returns:
            Number of articles updated
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("UPDATE articles SET views = 0")
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

# Convenience functions for direct use
def get_article_by_id(article_id: str, db_path: str = None) -> Optional[Dict[str, Any]]:
    """Get a single article by its ID."""
    db = ArticleDatabase(db_path)
    return db.get_article_by_id(article_id)

def get_articles_by_ids(article_ids: List[str], db_path: str = None) -> List[Dict[str, Any]]:
    """Get multiple articles by their IDs."""
    db = ArticleDatabase(db_path)
    return db.get_articles_by_ids(article_ids)

def search_articles(query: str, limit: int = 100, db_path: str = None) -> List[Dict[str, Any]]:
    """Search articles by title or abstract."""
    db = ArticleDatabase(db_path)
    return db.search_articles(query, limit)

def get_random_articles(limit: int = 10, db_path: str = None) -> List[Dict[str, Any]]:
    """Get random articles."""
    db = ArticleDatabase(db_path)
    return db.get_random_articles(limit)

def get_categories(db_path: str = None) -> List[Dict[str, Any]]:
    """Get all categories with article counts."""
    db = ArticleDatabase(db_path)
    return db.get_categories()

def get_article_stats(db_path: str = None) -> Dict[str, Any]:
    """Get comprehensive article statistics."""
    db = ArticleDatabase(db_path)
    return db.get_article_stats()

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "stats":
            stats = get_article_stats()
            print("Article Database Statistics:")
            print(f"Total articles: {stats['total_articles']}")
            print("\nArticles by category:")
            for cat in stats['by_category']:
                print(f"  {cat['category']}: {cat['count']}")
            print("\nTop subcategories:")
            for subcat in stats['by_subcategory']:
                print(f"  {subcat['subcategory']}: {subcat['count']}")
            print(f"\nArticles with entities: {stats['articles_with_entities']}")
            print(f"Articles with keywords: {stats['articles_with_keywords']}")
        
        elif command == "random":
            articles = get_random_articles(5)
            print("Random articles:")
            for article in articles:
                print(f"  {article['article_id']}: {article['title'][:60]}...")
        
        elif command == "search" and len(sys.argv) > 2:
            query = sys.argv[2]
            articles = search_articles(query, 5)
            print(f"Search results for '{query}':")
            for article in articles:
                print(f"  {article['article_id']}: {article['title'][:60]}...")
        
        else:
            print("Usage:")
            print("  python articles.py stats")
            print("  python articles.py random")
            print("  python articles.py search <query>")
    else:
        print("Usage:")
        print("  python articles.py stats")
        print("  python articles.py random")
        print("  python articles.py search <query>")

