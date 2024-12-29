# Use the 3.12 environment
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from LangGraph_agent import Agent

import json

app = FastAPI(title='ShopTalk Agent', openapi_url="/openapi.json")
api_router = APIRouter()

agent = Agent()

class MultimodalInput(BaseModel):
    image: str  # Base64-encoded string
    text: str  # Text prompt

@api_router.get("/", status_code=200)
def root() -> dict:
    """
    Root GET
    """
    return {"msg": "You can use this endpoint to interact with the chat-bot agent."}

@api_router.post("/prompt", status_code=201)
async def prompt(input: MultimodalInput) -> StreamingResponse:
    """
    Get agent response from input
    """
    async def streaming_generator():
        for message in agent.prompt(chat_id="1", prompt=input.text, image_b64=input.image):
            yield json.dumps(message) + "\n"
            
    return StreamingResponse(streaming_generator(), media_type="application/json")
    
app.include_router(api_router, prefix="/api/v1")
print('Agent endpoint up and running.')

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9001, log_level="debug")