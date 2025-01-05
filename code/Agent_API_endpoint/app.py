# Use the 3.12 environment
# import os

# os.environ["BLIP_2_MODEL"]="gs"
# os.environ["BLIP_2_PORT"]="9002"
# os.environ["CHROMA_PORT"]="8000"
# os.environ["CHROMA_MAX_IMAGES_PER_ITEM"]="21"
# os.environ["CHROMA_MAX_ITEMS"]="3"
# os.environ["OLLAMA_PORT"]="11434"
# os.environ["OLLAMA_MODEL"]="llama3.1:8b"
# os.environ["ABO_DIR_CONTAINER"]="/mnt/d/abo-dataset"


from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from LangGraph_agent import prompt, get_thread_ids, get_message_thread

import json

# from icecream import ic

app = FastAPI(title='ShopTalk Agent', openapi_url="/openapi.json")
api_router = APIRouter()

class PromptInput(BaseModel):
    thread_id: str  # for memory purposes
    image: str  # Base64-encoded string
    text: str  # Text prompt

@api_router.get("/", status_code=200)
async def root() -> dict:
    """
    Root GET
    """
    return {"msg": "You can use this endpoint to interact with the chat-bot agent."}

@api_router.post("/prompt", status_code=201)
async def prompt_fastapi(input: PromptInput) -> StreamingResponse:
    """
    Get agent response from input
    """
    def streaming_generator():
        for message in prompt(thread_id=input.thread_id, prompt_str=input.text, image_b64=input.image):
            yield json.dumps(message) + "\n"
            
    return StreamingResponse(streaming_generator(), media_type="application/json")

@api_router.get("/thread_id", status_code=200)
async def get_thread_ids_fastapi():
    """
    Get IDs of all chat threads
    """
    return {"thread_ids": get_thread_ids()}

@api_router.get("/thread_id/{thread_id}", status_code=200)
async def get_chat_history(thread_id: str):
    """
    Get a chat thread by ID
    """
    conversation, user_image = get_message_thread(thread_id)
    return {"messages": conversation, "user_image": user_image}

app.include_router(api_router, prefix="/api/v1")
print('Agent endpoint up and running.')

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9001, log_level="debug")