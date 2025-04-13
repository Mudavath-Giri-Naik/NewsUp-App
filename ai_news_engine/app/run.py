from fastapi import FastAPI
from app.routes import router  # ✅ absolute import
from dotenv import load_dotenv
import uvicorn
import os

# Load environment variables
load_dotenv()

app = FastAPI(
    title="NewsUp AI Engine",
    description="Extracts and classifies news articles from newspaper images.",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.run:app", host="0.0.0.0", port=port, reload=True)
