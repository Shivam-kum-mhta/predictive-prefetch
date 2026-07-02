from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .prediction import predict

# Create router for model-related endpoints
router = APIRouter(prefix="/model", tags=["model"])

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    user_history: List[str]
    top_k: Optional[int] = None

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_type: str
    status: str
    error: Optional[str] = None

@router.post("/predict", response_model=PredictionResponse)
def predict_articles(request: PredictionRequest):
    """
    Predict the next articles a user might read based on their reading history.
    
    Predictions are automatically enriched with article information before slider application.
    
    Args:
        request: PredictionRequest containing:
            - user_history: List of article IDs the user has read (most recent last)
            - top_k: Number of top predictions to return (optional, defaults to OUTPUT_ATMOST)
    
    Returns:
        PredictionResponse with predictions and metadata
    
    Example request:
    ```json
    {
        "user_history": ["N12345", "N67890"]
    }
    ```
    
    Example response:
    ```json
    {
        "predictions": [
            {
                "article_id": "N46394",
                "probability": 0.25,
                "category": "sports",
                "subcategory": "basketball_nba",
                "title": "LeBron James hires taco truck...",
                "abstract": "",
                "url": "https://...",
                "entities": [...],
                "keywords": [],
                "created_at": "2025-10-06 17:26:19"
            }
        ],
        "model_type": "sparse_markov_chain",
        "status": "success"
    }
    ```
    """
    try:
        # Convert Pydantic model to dict for the prediction function
        request_dict = request.dict()
        
        # Call the prediction function (enrichment now happens inside predict())
        result = predict(request_dict)
        #print("Result successful: ", result)
        
        # Check for errors in the result
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
        
        return PredictionResponse(**result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")