from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .articles import ArticleDatabase, get_article_stats, get_categories

# Create router for database-related endpoints
router = APIRouter(prefix="/articles", tags=["articles"])

# Pydantic model for enriched prediction
class EnrichedPrediction(BaseModel):
    article_id: str
    probability: float
    category: Optional[str] = None
    subcategory: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    url: Optional[str] = None
    entities: Optional[List[Dict[str, Any]]] = None
    keywords: Optional[List[Dict[str, Any]]] = None
    views: Optional[int] = 0
    created_at: Optional[str] = None

# Pydantic models for request/response validation
class ArticleResponse(BaseModel):
    article_id: str
    category: str
    subcategory: str
    title: str
    abstract: Optional[str]
    url: str
    entities: List[Dict[str, Any]]
    keywords: List[Dict[str, Any]]
    views: int = 0
    created_at: str

class ArticlesResponse(BaseModel):
    articles: List[ArticleResponse]
    total: int
    limit: int
    offset: int

class ArticleStatsResponse(BaseModel):
    total_articles: int
    total_views: int
    avg_views: float
    by_category: List[Dict[str, Any]]
    by_subcategory: List[Dict[str, Any]]
    articles_with_entities: int
    articles_with_keywords: int

class IncrementViewsRequest(BaseModel):
    article_ids: List[str]

# Initialize database
db = ArticleDatabase()

@router.get("/stats", response_model=ArticleStatsResponse)
def get_article_statistics():
    """
    Get comprehensive statistics about articles in the database.
    """
    try:
        stats = get_article_stats()
        return ArticleStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/categories")
def get_article_categories():
    """
    Get all article categories with counts.
    """
    try:
        categories = get_categories()
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/subcategories")
def get_article_subcategories(category: Optional[str] = Query(None, description="Filter by category")):
    """
    Get all article subcategories with counts.
    """
    try:
        subcategories = db.get_subcategories(category)
        return {"subcategories": subcategories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{article_id}", response_model=ArticleResponse)
def get_article(article_id: str):
    """
    Get a single article by its ID.
    """
    try:
        article = db.get_article_by_id(article_id)
        if not article:
            raise HTTPException(status_code=404, detail=f"Article {article_id} not found")
        
        # Convert datetime to string for JSON serialization
        if 'created_at' in article:
            article['created_at'] = str(article['created_at'])
        
        return ArticleResponse(**article)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=List[ArticleResponse])
def get_articles_batch(article_ids: List[str]):
    """
    Get multiple articles by their IDs.
    """
    try:
        if not article_ids:
            return []
        
        articles = db.get_articles_by_ids(article_ids)
        
        # Convert datetime to string for JSON serialization
        for article in articles:
            if 'created_at' in article:
                article['created_at'] = str(article['created_at'])
        
        return [ArticleResponse(**article) for article in articles]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/category/{category}")
def get_articles_by_category(
    category: str, 
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of articles to return")
):
    """
    Get articles by category.
    """
    try:
        articles = db.get_articles_by_category(category, limit)
        
        # Convert datetime to string for JSON serialization
        for article in articles:
            if 'created_at' in article:
                article['created_at'] = str(article['created_at'])
        
        return {
            "articles": [ArticleResponse(**article) for article in articles],
            "category": category,
            "count": len(articles),
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/subcategory/{subcategory}")
def get_articles_by_subcategory(
    subcategory: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of articles to return")
):
    """
    Get articles by subcategory.
    """
    try:
        articles = db.get_articles_by_subcategory(subcategory, limit)
        
        # Convert datetime to string for JSON serialization
        for article in articles:
            if 'created_at' in article:
                article['created_at'] = str(article['created_at'])
        
        return {
            "articles": [ArticleResponse(**article) for article in articles],
            "subcategory": subcategory,
            "count": len(articles),
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/query")
def search_articles(
    q: str = Query(..., description="Search query"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of articles to return")
):
    """
    Search articles by title or abstract.
    """
    try:
        if not q.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        articles = db.search_articles(q, limit)
        
        # Convert datetime to string for JSON serialization
        for article in articles:
            if 'created_at' in article:
                article['created_at'] = str(article['created_at'])
        
        return {
            "articles": [ArticleResponse(**article) for article in articles],
            "query": q,
            "count": len(articles),
            "limit": limit
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/random")
def get_random_articles(
    limit: int = Query(10, ge=1, le=100, description="Number of random articles to return")
):
    """
    Get random articles.
    """
    try:
        articles = db.get_random_articles(limit)
        
        # Convert datetime to string for JSON serialization
        for article in articles:
            if 'created_at' in article:
                article['created_at'] = str(article['created_at'])
        
        return {
            "articles": [ArticleResponse(**article) for article in articles],
            "count": len(articles),
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recent")
def get_recent_articles(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of recent articles to return")
):
    """
    Get the most recent articles.
    """
    try:
        articles = db.get_recent_articles(limit)
        
        # Convert datetime to string for JSON serialization
        for article in articles:
            if 'created_at' in article:
                article['created_at'] = str(article['created_at'])
        
        return {
            "articles": [ArticleResponse(**article) for article in articles],
            "count": len(articles),
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate")
def validate_article_ids(article_ids: List[str]):
    """
    Validate which article IDs exist in the database.
    """
    try:
        if not article_ids:
            return {"validated": {}}
        
        validation_results = db.validate_article_ids(article_ids)
        
        return {
            "validated": validation_results,
            "total_requested": len(article_ids),
            "total_valid": sum(validation_results.values()),
            "total_invalid": len(article_ids) - sum(validation_results.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def database_health_check():
    """
    Health check endpoint to verify the database is accessible.
    """
    try:
        # Try to get article count to verify database access
        count = db.get_article_count()
        return {
            "status": "healthy",
            "total_articles": count,
            "database_path": db.db_path
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.post("/increment-views")
def increment_article_views(request: IncrementViewsRequest):
    """
    Increment the view count for a list of articles.
    
    This endpoint should be called when users view articles to track popularity.
    
    Args:
        request: IncrementViewsRequest containing article_ids
    
    Returns:
        Number of articles updated
    
    Example request:
    {
        "article_ids": ["N12345", "N67890"]
    }
    """
    try:
        updated_count = db.increment_article_views(request.article_ids)
        return {
            "success": True,
            "updated_count": updated_count,
            "requested_count": len(request.article_ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/most-viewed")
def get_most_viewed_articles(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of articles to return")
):
    """
    Get the most viewed articles.
    
    Args:
        limit: Maximum number of articles to return
    
    Returns:
        List of most viewed articles with full details
    """
    try:
        articles = db.get_most_viewed_articles(limit)
        
        # Convert datetime to string for JSON serialization
        for article in articles:
            if 'created_at' in article:
                article['created_at'] = str(article['created_at'])
        
        return {
            "articles": [ArticleResponse(**article) for article in articles],
            "count": len(articles),
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/set-views/{article_id}")
def set_article_views(article_id: str, views: int = Query(..., ge=0, description="Number of views to set")):
    """
    Set the view count for a specific article.
    
    This is mainly for administrative purposes or testing.
    
    Args:
        article_id: Article ID
        views: Number of views to set
    
    Returns:
        Success status
    """
    try:
        success = db.set_article_views(article_id, views)
        if not success:
            raise HTTPException(status_code=404, detail=f"Article {article_id} not found")
        
        return {
            "success": True,
            "article_id": article_id,
            "views": views
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset-all-views")
def reset_all_article_views():
    """
    Reset all article views to 0.
    
    This is mainly for administrative purposes or testing.
    Use with caution as this affects all articles.
    
    Returns:
        Number of articles updated
    """
    try:
        updated_count = db.reset_all_views()
        return {
            "success": True,
            "updated_count": updated_count,
            "message": "All article views have been reset to 0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def enrich_predictions_with_articles(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Helper function to enrich predictions with full article information.
    
    Args:
        predictions: List of dicts with at least 'article_id' and 'probability' keys
    
    Returns:
        List of enriched predictions with full article information
    """
    if not predictions:
        return []
    
    # Extract article IDs from predictions
    article_ids = [pred.get("article_id") for pred in predictions if pred.get("article_id")]
    
    if not article_ids:
        return predictions
    
    # Fetch articles from database
    articles = db.get_articles_by_ids(article_ids)
    
    # Create a mapping from article_id to article data
    articles_map = {article["article_id"]: article for article in articles}
    
    # Enrich predictions with article data
    enriched = []
    for pred in predictions:
        article_id = pred.get("article_id")
        probability = pred.get("probability", 0.0)
        
        # Start with the prediction data
        enriched_pred = {
            "article_id": article_id,
            "probability": probability
        }
        
        # Add article data if available
        if article_id in articles_map:
            article = articles_map[article_id]
            enriched_pred.update({
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
        
        enriched.append(enriched_pred)
    
    return enriched

@router.post("/enrich-predictions", response_model=List[EnrichedPrediction])
def enrich_predictions_endpoint(predictions: List[Dict[str, Any]]):
    """
    Enrich a list of predictions with full article information.
    
    Args:
        predictions: List of predictions with at least 'article_id' and 'probability' keys
    
    Returns:
        List of enriched predictions with full article details
    
    Example request:
    [
        {"article_id": "N12345", "probability": 0.25},
        {"article_id": "N67890", "probability": 0.20}
    ]
    
    Example response:
    [
        {
            "article_id": "N12345",
            "probability": 0.25,
            "category": "sports",
            "subcategory": "basketball_nba",
            "title": "...",
            "abstract": "...",
            "url": "...",
            "entities": [...],
            "keywords": [...],
            "created_at": "..."
        }
    ]
    """
    try:
        enriched = enrich_predictions_with_articles(predictions)
        return [EnrichedPrediction(**pred) for pred in enriched]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
