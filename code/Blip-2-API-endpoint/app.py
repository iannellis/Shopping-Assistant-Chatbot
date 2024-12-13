# Use the 3.11 environment

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import pickle
import torch


app = FastAPI(title='BLIP-2 embeddings', openapi_url="/openapi.json")
api_router = APIRouter()

# the (eval) preprocessors for the model
with open('D:/pickles/processors.pkl', 'rb') as f:
    vis_processor, text_processor = pickle.load(f)

# the model itself
model = torch.load("D:/marqo-gs-10m/model-saves/pretrain_1epoch.pt")
model.eval()

class ImageInput(BaseModel):
    image: str  # Base64-encoded string
    
class TextInput(BaseModel):
    text: str  # Text input for embedding
    
class MultimodalInput(BaseModel):
    image: str  # Base64-encoded string
    text: str  # Text input for embedding
    
def decode_image(base64_str):
    """Decode a Base64 string into a PIL Image."""
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert('RGB')

@api_router.get("/", status_code=200)
def root() -> dict:
    """
    Root GET
    """
    return {"msg": "You can use this endpoint to get BLIP-2 embeddings."}

@api_router.post("/embed_image/", status_code=201)
async def create_recipe(*, input: ImageInput) -> dict:
    """
    Get image embeddings
    """
    image = decode_image(input.image)
    image_tensor = vis_processor(image)
    
    # Generate embedding
    sample = {'image': image_tensor}
    embedding = model.extract_features(sample, mode='image').image_embeds[0,0,:].detach().cpu().to_list()
    return {"embedding": embedding}

@api_router.post("/embed_text", status_code=201)
async def embed_text(input: TextInput):
    """
    Get text embeddings
    """
    text = text_processor(input.text)
    
    # Generate embedding
    sample = {'text_input': [text]}
    embedding = model.extract_features(sample, mode='text').text_embeds[0,0,:].detach().cpu().to_list()
    return {"embedding": embedding}

@api_router.post("/embed_multimodal", status_code=201)
async def embed_multimodal(input: MultimodalInput):
    """
    Get multimodal embeddings
    """
    image = decode_image(input.image)
    image_tensor = vis_processor(image)
    text = text_processor(input.text)
    
    # Generate embedding
    sample = {'image': image_tensor, 'text_input': [text]}
    embedding = model.extract_features(sample, mode='multimodal').multimodal_embeds[0,0,:].detach().cpu().to_list()
    return {"embedding": embedding}

app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="debug")