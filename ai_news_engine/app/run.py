# ai_news_engine/app/run.py

import uvicorn
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from .routes import router  # ✅ relative import

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="NewsUp AI Engine",
    description="Extracts and classifies news articles from newspaper images.",
    version="1.0.0"
)

# Include your routes
app.include_router(router)

# Run the server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("ai_news_engine.app.run:app", host="0.0.0.0", port=port, reload=True)
