import requests
import os

class Blip_2_Connection():
    def __init__(self):
        # Wait for BLIP-2 connection
        blip_2_port = os.environ["BLIP_2_PORT"]
        self.blip_2_url = "http://blip-2:" + blip_2_port + '/api/v1/'
        
        response = None
        while not response:
            try:
                response = requests.get(self.blip_2_url, timeout=5)
            except requests.ReadTimeout:
                print('Blip-2 endpoint does not appear to be running yet. Retrying')

    def blip_2_encode(self, image_b64: None, text: None) -> list | None:
        """Embed images and text using the BLIP-2 API endpoint"""
        if not image_b64 and not text:
            return
            
        if image_b64 and not text:
            response = requests.post(self.blip_2_url+'embed_image', json={'image': image_b64})  
        elif not image_b64 and text:
            response = requests.post(self.blip_2_url+'embed_text', json={'text': text})
        else:
            response = requests.post(self.blip_2_url+'embed_multimodal', json={'image': image_b64, 'text': text})
        
        return response.json()['embedding']