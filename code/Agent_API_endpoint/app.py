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
vectorstore = Chroma(collection_name='blip_2_'+blip_2_model, client=chroma_client)
retriever = vectorstore.as_retriever(search_kwargs = {"k": 3})

app = FastAPI(title='BLIP-2 embeddings', openapi_url="/openapi.json")
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