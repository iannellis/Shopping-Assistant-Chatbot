from langchain_ollama import ChatOllama
import chromadb
import numpy as np
import pandas as pd

import requests
import os
from httpx import ConnectError
from time import sleep
from collections import namedtuple
import base64

Image_Item_Pair_IDs = namedtuple('Image_Item_Pair_IDs', ['image_id', 'item_id'])
Image_Item_Pair_Data = namedtuple('Image_Item_Pair_Data', ['image_b64', 'item_str'])

class Chroma_Collection_Connection():
    """Class to connect to the Chroma database and query it with embeddings, then filter
    the results down to CHROMA_MAX_ITEMS items."""
    def __init__(self):
        """Setup Chroma DB connection. Also makes sure database is loaded into RAM."""
        chroma_port = int(os.environ["CHROMA_PORT"])
        blip_2_model = os.environ["BLIP_2_MODEL"]
        max_images_per_item = os.environ["CHROMA_MAX_IMAGES_PER_ITEM"]
        max_items = os.environ["CHROMA_MAX_ITEMS"]
        n_return = max_items * max_images_per_item
        
        client = None
        while not client:
            try:
                client = chromadb.HttpClient(host='chroma', port=chroma_port)
            except ValueError:
                print('Chroma DB does not appear to be running yet. Retrying.')
                sleep(2)
        
        embedding_len = 768
        embedding_test = [1] * embedding_len
        collection = client.get_collection(name='blip_2_'+blip_2_model)
        _ = collection.query(query_embeddings=[embedding_test], include=["metadatas", "distances"], n_results=n_return)
        
        self.collection = collection
        self.max_return_items = max_items
        self.n_return = self.n_return
        
        self.blip_2_connection = Blip_2_Connection()
    
    def query_image_text(self, image_b64: None, text: None) -> dict:
        """Get the image_ids and item_ids of the top CHROMA_MAX_ITEMS products matching
        the query."""
        embeddings = self.blip_2_connection.embed(image_b64=image_b64, text=text)
        raw_return = self._query_embeddings(embeddings)   
        return self._filter(raw_return)
    
    def _query_embeddings(self, embeddings: np.array) -> dict:
        """Get the top n_return image_ids and item_ids matching a query's embeddings."""
        return self.collection.query(query_embeddings=[embeddings], include=["metadatas", "distances"], n_results=self.n_return)   

    def _filter(self, query_results: dict) -> list[tuple]:
        """Filter n_return image_ids and item_ids down to CHROMA_MAX_ITEMS items."""
        seen_items = set()
        image_item_pairs = []
        metadatas = query_results['metadatas']
        
        i = 0
        while len(image_item_pairs) < self.max_return_items and i < len(metadatas):
            metadatum = metadatas[i]
            i+=1
            
            item_id = metadatum['item_id']
            if item_id not in seen_items:
                seen_items.add(item_id)
                if 'image_id' in metadatum:
                    image_item_pair = Image_Item_Pair_IDs(image_id=metadatum['image_id'], item_id=item_id)
                else:
                    image_item_pair = Image_Item_Pair_IDs(image_id=None, item_id=item_id)
                image_item_pairs.append(image_item_pair)
        return image_item_pairs

class Blip_2_Connection():
    """Class to manage the connection to the BLIP-2 model and call it to get embeddings
    for data."""
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

    def embed(self, image_b64: None, text: None) -> list | None:
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

def connect_ollama_llm():
    """Setup Ollama connection. Also makes sure model's loaded into video RAM. Returns
    the model for use by LangGraph."""
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
    return llm

class ABO_Dataset():
    """Class to load the ABO dataset metadata, perform operations on it, and load images
    when required."""
    def __init__(self):
        abo_fname = "abo-listings-final-draft.pkl"
        abo_dir = os.environ["ABO_DIR_CONTAINER"]
        self.abo_meta = self._load_metadata(abo_dir + '/' + abo_fname)
        self.abo_image_meta = pd.read_csv(abo_dir + '/images/metadata/images.csv').set_index('image_id')
        self.abo_image_dir = abo_dir + '/images/small/'
    
    def _load_metadata(self, fpath):
        abo_meta = pd.read_pickle(fpath)
        abo_meta = abo_meta[['item_name', 'brand', 'model_name', 'model_year',
                            'product_description', 'product_type', 'color',
                            'fabric_type', 'style', 'material', 'pattern', 
                            'finish_type', 'country', 'marketplace', 'domain_name',
                            'item_keywords', 'bullet_point']]
        return abo_meta
    
    def _get_b64_encoded_image(self, image_id: str):
        fpath = self.abo_image_dir + self.abo_image_meta.loc[image_id, 'path']
        with open(fpath, 'rb') as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes)
    
    def _get_item_as_str(self, item_id: str):
        row = self.abo_meta[item_id]
        row_filtered = row.dropna()
        text = []
        for row_heading, row_item in zip(row_filtered.index, row_filtered):
            text.append('{')
            text.append(row_heading.replace('_', ' '))
            text.append(': ')
            text.append(str(row_item))
            text.append('}')
            text.append('; ')
        return ''.join(text)
    
    def get_image_item_pairs_data(self, pairs: list[Image_Item_Pair_IDs]) -> list[Image_Item_Pair_Data]:
        """Loop through the image_id and item_id pairs and get the image incoded in b64
        and the item data as a string."""
        image_item_pairs_data = []
        for image_id, item_id in pairs:
            image_b64 = self._get_b64_encoded_image(image_id) if image_id else None
            item_str = self._get_item_as_str(item_id)
            image_item_pairs_data.append(Image_Item_Pair_Data(image_b64=image_b64, item_str=item_str))
        return image_item_pairs_data
                