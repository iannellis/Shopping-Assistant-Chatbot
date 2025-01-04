import requests
import streamlit as st
from PIL import Image
import io
import json
import base64

from icecream import ic

# agent_url = "http://agent:" + os.environ["AGENT_PORT"] + "/api/v1/"
agent_url = "http://localhost:9001/api/v1"

# ----------------------------Functons used later---------------------------------------
def new_chat_name():
    """Clear session state when user enters a new chat name"""
    thread_id = st.session_state.chat_name_input.strip().replace(" ", "_")
    if  thread_id in st.session_state.all_thread_ids:
        st.sidebar.warning("Chat name already in use.")
    else:
        st.session_state.thread_name = thread_id
        st.session_state.messages = []
        st.session_state.user_image = None
        st.session_state.user_image_sent = False
    st.session_state.chat_name_input = ""

def stream_response(response):
    """Process the streamed JSON objects from the agent"""
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

def integrate_images(messages):
    """When loading an existing thread, integrate the images sent from the agent
    into a form usable by Streamlit."""
    for message in messages:
        if message["images"]:
            new_message_content = []
            images = message.pop("images")
            for idx, base64_image in enumerate(images):
                new_message_content.append("### Product " + str(idx+1) + ":")
                image = Image.open(io.BytesIO(base64.b64decode(base64_image)))
                new_message_content.append(image)
            new_message_content.append(message["content"])
            message["content"] = new_message_content
    return messages

# ----------------------------------The UI--=======-------------------------------------

st.title('ShopTalk Chatbot 🤖')

# Initialize chat history and related states
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
if "user_image" not in st.session_state:
    st.session_state["user_image"] = None
    
if "user_image_sent" not in st.session_state:
    st.session_state["user_image_sent"] = False
    
if "thread_name" not in st.session_state:
    st.session_state["thread_name"] = "First_Chat"

# Sidebar for chat name input and listing previous chats
st.sidebar.title("Your Chats")

# retrieve previous chat threads
if "all_thread_ids" not in st.session_state:
    response = requests.get(agent_url + '/thread_id')
    st.session_state["all_thread_ids"] = response.json()['thread_ids']

# If a user inputs a new chat name, reset the chat session (calls new_chat_name)
st.sidebar.text_input("Enter chat name to start a new chat:", key="chat_name_input", on_change=new_chat_name)

# populate sidebar buttons and handle click
for thread_id in reversed(st.session_state.all_thread_ids):
    # button push or page refresh (changes to First Chat in latter case)
    if st.sidebar.button(thread_id.replace("_", " ")) or thread_id==st.session_state.thread_name \
            and not st.session_state.messages:
        chat_history = requests.get(agent_url + '/thread_id/' + thread_id).json()
        st.session_state.thread_name = thread_id
        st.session_state.messages = integrate_images(chat_history["messages"])
        st.session_state.user_image = chat_history["user_image"]
        if st.session_state.user_image:
            st.session_state.user_image_sent = True

# Keep thread id near the top of the chat 
st.subheader(f"Chat Name: {st.session_state.thread_name.replace("_", " ")}")

# Show file uploader if no file has been uploaded
if not st.session_state.user_image:
    uploaded_file = st.file_uploader("Upload an image of the product you're searching for:")
    if uploaded_file is not None:
        image = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
        st.session_state.user_image = image
        st.success("File uploaded successfully!")

# Show the file after it is uploaded without the option to upload an image
if st.session_state.user_image:
    st.write("### Your image")
    image = Image.open(io.BytesIO(base64.b64decode(st.session_state.user_image)))
    st.write(image)
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], list):
            st.write_stream(message["content"])
        else:
            st.write(message["content"])

# React to user input
if prompt := st.chat_input("What are you shopping for?"):
    # If this thread was not previously tracked, it now is.
    if not st.session_state.messages:
        st.session_state.all_thread_ids.append(st.session_state.thread_name)
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
        
    # send to agent
    thread_id = st.session_state.thread_name
    to_post = {"thread_id": thread_id, "text": prompt, "image": ''}
    if st.session_state.user_image and not st.session_state.user_image_sent:
        to_post["image"] = st.session_state.user_image
        st.session_state.user_image_sent = True
    response = requests.post(agent_url + '/prompt', json=to_post, stream=True)
    
    # stream to user
    with st.chat_message("assistant"):
        streamed_response = st.write_stream(stream_response(response))
    
    # Add agent message to chat history
    st.session_state.messages.append({"role": "assistant", "content": streamed_response})
    st.rerun() # so thread_id button shows up if it's the first prompt
               # also removes upload option if image just provided


