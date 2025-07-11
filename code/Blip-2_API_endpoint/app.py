# Use the 3.11 environment

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
import os
import base64
from io import BytesIO
from PIL import Image
import pickle
import torch

import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title='BLIP-2 embeddings', openapi_url="/openapi.json")
api_router = APIRouter()

# the (eval) preprocessors for the model
model_selection = os.environ["BLIP_2_MODEL"]
if model_selection == 'coco':
    with open('blip-2-processors-coco.pkl', 'rb') as f:
        vis_processor, text_processor = pickle.load(f)
else:
    with open('blip-2-processors-pretrain.pkl', 'rb') as f:
        vis_processor, text_processor = pickle.load(f)

# the model itself
print('Loading BLIP-2 model ' + model_selection)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models_dir = os.environ["BLIP_2_DIR_CONTAINER"]
model = torch.load(models_dir + "/blip-2-" + model_selection + ".pt")
model.to(device)
model.eval()
print('Done loading BLIP-2 model.')

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

@api_router.post("/embed_image", status_code=201)
async def embed_image(input: ImageInput) -> dict:
    """
    Get image embeddings
    """
    image = decode_image(input.image)
    image_tensor = vis_processor(image).unsqueeze(0).to(device)
    
    # Generate embedding
    sample = {'image': image_tensor}
    embedding = model.extract_features(sample, mode='image').image_embeds[0,:4,:].detach().cpu().numpy()
    return {"embedding": embedding.reshape(-1).tolist()}

@api_router.post("/embed_text", status_code=201)
async def embed_text(input: TextInput):
    """
    Get text embeddings
    """
    text = text_processor(input.text)
    
    # Generate embedding
    sample = {'text_input': [text]}
    embedding = model.extract_features(sample, mode='text').text_embeds[0,:2,:].detach().cpu().numpy()
    return {"embedding": embedding.reshape(-1).tolist()}

@api_router.post("/embed_multimodal", status_code=201)
async def embed_multimodal(input: MultimodalInput):
    """
    Get multimodal embeddings
    """
    image = decode_image(input.image)
    image_tensor = vis_processor(image).unsqueeze(0).to(device)
    text = text_processor(input.text)
    
    # Generate embedding
    sample = {'image': image_tensor, 'text_input': [text]}
    embedding = model.extract_features(sample, mode='multimodal').multimodal_embeds[0,:4,:].detach().cpu().numpy()
    return {"embedding": embedding.reshape(-1).tolist()}

app.include_router(api_router, prefix="/api/v1")
print('BLIP-2 endpoint up and running.')

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9002, log_level="debug")