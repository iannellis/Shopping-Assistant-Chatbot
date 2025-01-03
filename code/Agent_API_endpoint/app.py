# Use the 3.12 environment
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from LangGraph_agent import prompt, get_thread_ids, retrieve_message_thread

import json

app = FastAPI(title='ShopTalk Agent', openapi_url="/openapi.json")
api_router = APIRouter()

class MultimodalInput(BaseModel):
    chat_id: str  # for memory purposes
    image: str  # Base64-encoded string
    text: str  # Text prompt

@api_router.get("/", status_code=200)
async def root() -> dict:
    """
    Root GET
    """
    return {"msg": "You can use this endpoint to interact with the chat-bot agent."}

@api_router.post("/prompt", status_code=201)
async def prompt_fastapi(input: MultimodalInput) -> StreamingResponse:
    """
    Get agent response from input
    """
    async def streaming_generator():
        for message in prompt(chat_id=input.chat_id, prompt=input.text, image_b64=input.image):
            yield json.dumps(message) + "\n"
            
    return StreamingResponse(streaming_generator(), media_type="application/json")

@api_router.get("/thread_id", status_code=200)
async def get_thread_ids():
    """
    Get IDs of all chat threads
    """
    return {"thread_ids": get_thread_ids()}

@app.get("/thread_ids/{thread_id}", status_code=200)
async def get_chat_history(thread_id: str):
    """
    Get a chat thread by ID
    """
    conversation, user_image = retrieve_message_thread(thread_id)
    return {"messages": conversation, "user_image": user_image}

app.include_router(api_router, prefix="/api/v1")
print('Agent endpoint up and running.')

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn
    import os
    
    os.environ["BLIP_2_MODEL"]="gs"
    os.environ["BLIP_2_PORT"]="9002"
    os.environ["CHROMA_PORT"]="8000"
    os.environ["CHROMA_MAX_IMAGES_PER_ITEM"]="21"
    os.environ["CHROMA_MAX_ITEMS"]="3"
    os.environ["OLLAMA_PORT"]="11434"
    os.environ["OLLAMA_MODEL"]="llama3.1:8b"
    os.environ["ABO_DIR_CONTAINER"]="/mnt/d/abo-dataset"

    uvicorn.run(app, host="0.0.0.0", port=9001, log_level="debug")