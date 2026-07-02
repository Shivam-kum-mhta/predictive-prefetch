from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models.api import router as model_router
from app.db.api import router as db_router

app = FastAPI(
    title="Predictive Prefetch API", 
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(model_router)
app.include_router(db_router)

@app.get("/")
def read_root():
    return {
        "message": "Predictive Prefetch API", 
        "version": "1.0.0",
        "endpoints": {
            "model": {
                "predict": "/model/predict (supports ?enrich=true/false)",
                "info": "/model/info", 
                "health": "/model/health"
            },
            "articles": {
                "stats": "/articles/stats",
                "categories": "/articles/categories",
                "search": "/articles/search/query",
                "random": "/articles/random",
                "recent": "/articles/recent",
                "most_viewed": "/articles/most-viewed",
                "increment_views": "/articles/increment-views (POST)",
                "enrich_predictions": "/articles/enrich-predictions",
                "health": "/articles/health"
            },
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
def health_check():
    """
    Global health check endpoint.
    """
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "message": "Predictive Prefetch API is running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9090, reload=True)