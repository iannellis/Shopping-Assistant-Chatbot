from fastapi import FastAPI, APIRouter
from fastapi.responses import StreamingResponse
import asyncio

import json
import base64

with open('assets/sofa.jpg', "rb") as f:
    img = base64.b64encode(f.read()).decode("utf-8")
    
app = FastAPI(title='Image and text streamer', openapi_url="/openapi.json")
api_router = APIRouter()

@api_router.get("/", status_code=200)
async def root() -> dict:
    encoded_images = [
        img,
    ]

    async def streaming_generator():
        # Send all images as the first chunk
        pre_json_line = {"images": encoded_images, "text": ""}
        yield json.dumps(pre_json_line) + "\n"
        # Send subsequent text tokens
        tokens = ["Hello, ", "world! ", "This ", "is ", "a ", "streaming", " ", "response."]
        for token in tokens:
            yield f'{{"images": [], "text": "{token}"}}\n'
            await asyncio.sleep(0.5)

    return StreamingResponse(streaming_generator(), media_type="application/json")

app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8500, log_level="debug")