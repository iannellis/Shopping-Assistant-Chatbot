# Use the 3.12 environment
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama

from httpx import ConnectError
import chromadb
from time import sleep
import requests

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
import os
import base64

# setup Ollama connection
model_name = os.environ["OLLAMA_MODEL"]
ollama_port = os.environ["OLLAMA_PORT"]
llm = ChatOllama(base_url = "ollama:"+ollama_port, model = model_name)
response = None
# Check if Ollama is up and running and load the model into memory
while not response:
    try:
        response = llm.invoke('hello')
    except ConnectError:
        print('Ollama does not appear to be running yet. Retrying.')

# setup Chroma DB connection
chroma_port = int(os.environ["CHROMA_PORT"])
chroma_client = None
while not chroma_client:
    try:
        chroma_client = chromadb.HttpClient(host='chroma', port=chroma_port)
    except ValueError:
        print('Chroma DB does not appear to be running yet. Retrying.')
        sleep(2)

blip_2_model = os.environ["BLIP_2_MODEL"]
max_images_per_item = os.environ["CHROMA_MAX_IMAGES_PER_ITEM"]
max_items = os.environ["CHROMA_MAX_ITEMS"]
vectorstore = Chroma(collection_name='blip_2_'+blip_2_model, client=chroma_client)
retriever = vectorstore.as_retriever(search_kwargs = {"k": max_items * max_images_per_item})

# Wait for BLIP-2 connection
blip_2_port = os.environ["BLIP_2_PORT"]
blip_2_url = "http://blip-2:" + blip_2_port + '/api/v1/'
response = None
while not response:
    try:
        response = requests.get(blip_2_url, timeout=5)
    except requests.ReadTimeout:
        print('Blip-2 endpoint does not appear to be running yet. Retrying')


def encode_image(image: bytes) -> str:
    """Encode raw image using base64"""
    return base64.b64encode(image)

def blip_2_encode(image: None, text: None) -> list | None:
    """Embed images and text using the BLIP-2 API endpoint"""
    if not image and not text:
        return
    
    if image:
        encoded_image = encode_image(image)
    
    if image and not text:
        response = requests.post(blip_2_url+'embed_image', json={'image': encoded_image})  
    elif not image and text:
        response = requests.post(blip_2_url+'embed_text', json={'text': text})
    else:
        response = requests.post(blip_2_url+'embed_multimodal', json={'image': encode_image, 'text': text})
    
    return response.json()['embedding']

app = FastAPI(title='ShopTalk Agent', openapi_url="/openapi.json")
api_router = APIRouter()

@api_router.get("/", status_code=200)
def root() -> dict:
    """
    Root GET
    """
    return {"msg": "You can use this endpoint to interact with the chat-bot agent."}

app.include_router(api_router, prefix="/api/v1")
print('Agent endpoint up and running.')

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9001, log_level="debug")