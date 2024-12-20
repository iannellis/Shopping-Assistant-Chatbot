import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, sys
from typing import List, Dict


app = FastAPI()

class ChatQuery(BaseModel):
    name: str
    price: float 

class ChatData(BaseModel):
    model: str
    messages : List[Dict[str, str]]
    
    
@app.get("/hello")
def read_root():
    return {"name": 'hello', "price": 10.99}

@app.post("/chat/")
def llm_query(chat: ChatData):
    print('chat from api: ', chat)
    return "response from model "

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8010, reload=True)