# Use the 3.12 environment

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
import os
import base64

app = FastAPI(title='BLIP-2 embeddings', openapi_url="/openapi.json")
api_router = APIRouter()

@api_router.get("/", status_code=200)
def root() -> dict:
    """
    Root GET
    """
    return {"msg": "You can use this endpoint to interact with the LLM."}

app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9001, log_level="debug")