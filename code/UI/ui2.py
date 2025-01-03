import requests
import streamlit as st
from PIL import Image
import io
import json
import base64
import os

agent_url = "http://agent:" + os.environ["AGENT_PORT"] + "/api/v1/"

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

# Initialize chat history and related states
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
if "user_image" not in st.session_state:
    st.session_state["user_image"] = None
    
if "user_image_sent" not in st.session_state:
    st.session_state["user_image_sent"] = False

# Sidebar for chat name input and listing previous chats
st.sidebar.title("Chat Manager")

# retrieve previous chat threads
if "all_chat_ids" not in st.session_state:
    st.session_state["all_chat_ids"] = requests.get(agent_url + '/chat_id')['chat_ids']

if "chat_name" not in st.session_state:
    st.session_state["chat_name"] = "New Chat"

# If a user inputs a new chat name, reset the chat session
if chat_name_input := st.sidebar.text_input("Enter chat name to start a new chat:").strip():
    chat_name_input = chat_name_input.replace(" ", "_")
    if  chat_name_input in st.session_state.all_chat_ids:
        st.sidebar.warning("Chat name already in use.")
    else:
        st.session_state.all_chat_ids.append(chat_name_input)
        st.session_state.chat_name = chat_name_input
        st.session_state.messages = []
        st.session_state.user_image = None
        st.session_state.user_image_sent = False

st.subheader(f"Chat Name: {st.session_state.chat_name.replace("_", " ")}")

# populate sidebar buttons and handle click
for chat_id in reversed(st.session_state.all_chat_ids):
    if st.sidebar.button(chat_id.replace("_", " ")):
        chat_history = requests.get(agent_url + '/chat_id/' + chat_id)['chat']
        st.session_state.chat_name = chat_id
        st.session_state.messages = chat_history["messages"]
        st.session_state.user_image = chat_history["user_image"]
        st.session_state.user_image_sent = True

# Show file uploader if no file has been uploaded
if not st.session_state["user_image"]:
    uploaded_file = st.file_uploader("Upload an image of the product you're searching for:")
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        st.session_state["user_image"] = image
        st.success("File uploaded successfully!")
        st.rerun()

# if not st.session_state.messages and st.session_state.chat_name:

# Show the file after it is uploaded without the option to upload an image
if st.session_state["user_image"]:
    st.write("Your image")
    st.write(st.session_state["user_image"])
    
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
    chat_id = st.session_state.chat_name
    to_post = {"chat_id": chat_id, "text": prompt}
    if st.session_state["user_image"] and not st.session_state["user_image_sent"]:
        to_post["image"] = st.session_state["user_image"]
        st.session_state["user_image_sent"] = True

    response = requests.post(agent_url + '/prompt', data=to_post, stream=True)
    # stream to user
    with st.chat_message("assistant"):
        streamed_response = st.write_stream(stream_response(response))
    
    # Add agent message to chat history
    st.session_state.messages.append({"role": "assistant", "content": streamed_response})
    


