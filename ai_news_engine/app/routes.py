# app/routes.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.classifier import predict_category
from app.vision import extract_articles_from_images

router = APIRouter()

@router.get("/process")
def process_articles():
    articles = extract_articles_from_images()

    # Add category prediction
    for article in articles:
        article["category"] = predict_category(article["description"])

    return JSONResponse(content={"status": "success", "articles": articles})
