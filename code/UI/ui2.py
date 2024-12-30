import requests
import streamlit as st
from PIL import Image
import io
import json
import base64
from icecream import ic
import os

agent_endpoint = "http://agent:" + os.environ["AGENT_PORT"] + "/api/v1/prompt"

def stream_response(response):
    # Process the streamed JSON objects
    for chunk in response.iter_lines(decode_unicode=True):
        if chunk:  # Skip empty lines
            try:
                # Parse the JSON object
                data = json.loads(chunk)
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


st.title('ShopTalk Chatbot 🤖')

if "user_image" not in st.session_state:
    st.session_state["user_image"] = None
    
if "image_sent" not in st.session_state:
    st.session_state["image_sent"] = False

# Show file uploader if no file has been uploaded
if not st.session_state["user_image"]:
    uploaded_file = st.file_uploader("Upload an image of the product you're searching for:")
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        st.session_state["user_image"] = image
        st.success("File uploaded successfully!")
        st.rerun()

# Show the file after it is uploaded without the option to upload an image
if st.session_state["user_image"]:
    st.write("Your image")
    st.write(st.session_state["user_image"])
    
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# React to user input
if prompt := st.chat_input("What are you shopping for?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
        
    # send to agent
    to_post = {"text": prompt}
    if st.session_state["user_image"] and not st.session_state["image_sent"]:
        to_post["image"] = st.session_state["user_image"]
        st.session_state["image_sent"] = True
    response = requests.post(agent_endpoint, data=to_post, stream=True)
    # stream to user
    with st.chat_message("assistant"):
        streamed_response = st.write_stream(stream_response(response))
    
    # Add agent message to chat history
    st.session_state.messages.append({"role": "assistant", "content": stream_response})