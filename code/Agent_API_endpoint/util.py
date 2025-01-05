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

from ollama import Client

Item_Data = namedtuple('Image_Item_Pair_Data', ['image_b64', 'item_str'])

CHROMA_HOST = "chroma"
OLLAMA_HOST = "ollama"
BLIP_2_HOST = "blip-2"

# CHROMA_HOST = "localhost"
# OLLAMA_HOST = "localhost"
# BLIP_2_HOST = "localhost"

class Chroma_Collection_Connection():
    """Class to connect to the Chroma database and query it with embeddings, then filter
    the results down to CHROMA_MAX_ITEMS items. Includes both the BLIP-2 and text-only
    collections. Probably should be separated if we were to tidy things up."""
    def __init__(self):
        """Setup Chroma DB connection. Also makes sure database is loaded into RAM."""
        chroma_port = int(os.environ["CHROMA_PORT"])
        blip_2_model = os.environ["BLIP_2_MODEL"]
        max_images_per_item = int(os.environ["CHROMA_MAX_IMAGES_PER_ITEM"])
        max_items = int(os.environ["CHROMA_MAX_ITEMS"])
        n_return = max_items * max_images_per_item
        
        # wait for Chroma to come online
        client = None
        while not client:
            try:
                client = chromadb.HttpClient(host=CHROMA_HOST, port=chroma_port)
            except ValueError:
                print('Chroma DB does not appear to be running yet. Retrying.')
                sleep(2)
        
        embedding_len = 768 * 4
        embedding_test = [1] * embedding_len
        multimodal_collection_name = 'blip_2_'+blip_2_model+'_multimodal'
        print('Using Chroma collection ' + multimodal_collection_name)
        collection_multimodal = client.get_collection(name=multimodal_collection_name)
        _ = collection_multimodal.query(query_embeddings=[embedding_test], include=["metadatas"], n_results=n_return)

        collection_text = client.get_collection(name='text_only')

        self.collection_multimodal = collection_multimodal
        self.collection_text = collection_text
        self.max_return_items = max_items
        self.n_return = n_return
        
        self.blip_2_connection = Blip_2_Connection()
        
    def query_image_text(self, image_b64=None, text=None) -> dict:
        """Get the image_ids and item_ids of the top CHROMA_MAX_ITEMS products matching
        the query."""
        embeddings = self.blip_2_connection.embed(image_b64=image_b64, text=text)
        raw_return = self._query_multimodal_embeddings(embeddings)   
        return self._filter_multimodal(raw_return)
    
    def query_text(self, text=None) -> list[str]:
        """Get the text-only documents of the top CHROMA_MAX_ITEMS matching the query."""
        query_return = self.collection_text.query(query_texts=[text], include=[], n_results=self.max_return_items)
        return query_return['ids'][0]
    
    def _query_multimodal_embeddings(self, embeddings: np.array) -> dict:
        """Get the top n_return image_ids and item_ids matching a query's embeddings."""
        return self.collection_multimodal.query(query_embeddings=[embeddings], include=["metadatas"], n_results=self.n_return)   

    def _filter_multimodal(self, query_results: dict) -> list[tuple]:
        """Filter n_return image_ids and item_ids down to CHROMA_MAX_ITEMS items."""
        items = []
        metadatas = query_results['metadatas'][0]
        
        i = 0
        while len(items) < self.max_return_items and i < len(metadatas):
            metadatum = metadatas[i]
            i+=1
            item_id = metadatum['item_id']
            if item_id not in items:
                items.append(item_id)
        return items

class Blip_2_Connection():
    """Class to manage the connection to the BLIP-2 model and call it to get embeddings
    for data."""
    def __init__(self):
        # Wait for BLIP-2 connection
        blip_2_port = os.environ["BLIP_2_PORT"]
        self.blip_2_url = "http://" + BLIP_2_HOST + ":" + blip_2_port + '/api/v1/'
        
        response = None
        while not response:
            try:
                response = requests.get(self.blip_2_url, timeout=5)
            except requests.exceptions.ConnectionError:
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
    ollama_url = "http://" + OLLAMA_HOST + ":" + ollama_port
    
    # make sure model is pulled
    ollama_client = Client(host=ollama_url)
    ollama_client.pull(model_name)
    
    llm = ChatOllama(base_url=ollama_url, model=model_name)
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
        self.abo_meta, self.abo_product_images = self._load_metadata(abo_dir + '/' + abo_fname)
        self.abo_image_meta = pd.read_csv(abo_dir + '/images/metadata/images.csv').set_index('image_id')
        self.abo_image_dir = abo_dir + '/images/small/'
    
    def _load_metadata(self, fpath):
        abo_meta = pd.read_pickle(fpath)
        abo_product_images = abo_meta[['main_image_id', 'other_image_id']]
        
        # leave out item_keywords because it can have a rediculous number of elements and confuses the LLM
        abo_meta = abo_meta[['item_name', 'brand', 'model_name', 'model_year',
                            'product_description', 'product_type', 'color',
                            'fabric_type', 'style', 'material', 'pattern', 
                            'finish_type', 'country', 'marketplace', 'domain_name',
                            'bullet_point']]
        abo_meta = abo_meta.rename(lambda x: str.replace(x, '_', ' '), axis='columns')
        # This one confuses the LLM. It thinks it's country of origin.
        abo_meta = abo_meta.rename(columns={'country': 'country of marketplace'})
        return abo_meta, abo_product_images
    
    def get_items_data(self, item_ids: list) -> list[Item_Data]:
        """Loop through the item_ids and get the image incoded in b64
        and the item data as a string."""
        items_data = []
        for item_id in item_ids:
            main_image_id = self.abo_product_images.loc[item_id, 'main_image_id']
            other_image_id = self.abo_product_images.loc[item_id, 'other_image_id']
            if main_image_id:
                image_b64 = self._get_b64_encoded_image(main_image_id)
            elif other_image_id:  # silly, but does occur
                image_b64 = self._get_b64_encoded_image(other_image_id[0])
            else:
                image_b64 = None
            
            item_str = self._get_item_as_str(item_id)
            items_data.append(Item_Data(image_b64=image_b64, item_str=item_str))
        return items_data    
    
    def _get_b64_encoded_image(self, image_id: str):
        """Encode the image as b64 for passing around via LangGraph."""
        fpath = self.abo_image_dir + self.abo_image_meta.loc[image_id, 'path']
        with open(fpath, 'rb') as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode("utf-8")
    
    def _get_item_as_str(self, item_id: str):
        """Get the row of data and convert it to a string consumable by an LLM."""
        row = self.abo_meta.loc[item_id]
        row_filtered = row.dropna()
        text = []
        for row_heading, row_item in zip(row_filtered.index, row_filtered):
            if row_item:  # some are things like empty lists
                text.append('{')
                text.append(row_heading)
                text.append(': ')
                text.append(str(row_item).replace('\n', ' ').replace('^', ' ').replace(',', ', '))
                text.append('}')
                text.append('; ')
        return ''.join(text)