import requests
from PIL import Image
import io
import json
import base64
import os
import time
import numpy as np

initial_chunk_time = ""

AGENT_URL = os.getenv("AGENT_URL")

if ( AGENT_URL is None or AGENT_URL == ""):   
    print("AGENT_URL is not set. Please set the AGENT_URL environment variable.")
    exit(1)



def stream_response(response):
    """Process the streamed JSON objects from the agent"""
    start_time = time.perf_counter()
    end_time = None
    is_initial_chunk = True
    initial_chunk_time = ""

    for chunk in response.iter_lines(decode_unicode=True):
        if chunk:  # Skip empty lines
            try:
                # Parse the JSON object
                data = json.loads(chunk)
                if data.get("images") or data.get("text"):
                    if is_initial_chunk:
                        end_time = time.perf_counter()
                        is_initial_chunk = False
                        initial_chunk_time = end_time - start_time
                        yield initial_chunk_time

                    if data.get("images"):
                        for idx, base64_image in enumerate(data["images"]):
                            if base64_image:
                                yield "### Product " + str(idx+1) + ":"
                                yield Image.open(io.BytesIO(base64.b64decode(base64_image)))
                    if data.get("text"):
                        # Handle text tokens. Need 2 spaces before every new line for streamlit.
                        yield data['text'].replace('\n', '  \n')

            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")

    yield initial_chunk_time

agent_url = AGENT_URL + "/prompt"  # add prompt path

# Array of thread_id and text messages
requests_data = [
    {"thread_id": "1400", "text": "give me red shoes"},
    {"thread_id": "1401", "text": "show me blue shirts"},
    {"thread_id": "1402", "text": "find green pants"}
]

headers = {
    "Content-Type": "application/json"
}

initial_chunk_times = []

for data in requests_data:
    to_post = {"thread_id": data["thread_id"], "text": data["text"], "image": ''}
    print(f"Posting to URL: {agent_url} with payload: {to_post}")

    try:
        response = requests.post(agent_url, headers=headers, json=to_post, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        streamed_response = stream_response(response)
        
        for item in streamed_response:
            if isinstance(item, float):
                initial_chunk_times.append(item)
            print(item)  # Print each item in the streamed response

    except requests.RequestException as e:
        print(f"Request failed: {e}")

# Calculate and print mean and 95th percentile of initial_chunk_times
if initial_chunk_times:
    mean_time = np.mean(initial_chunk_times)
    percentile_95_time = np.percentile(initial_chunk_times, 95)
    print(f"\nMean initial chunk time: {mean_time} seconds")
    print(f"95th percentile initial chunk time: {percentile_95_time} seconds")
else:
    print("\nNo initial chunk times recorded.")