import streamlit as st
#from openai import OpenAI
import requests

url="http://localhost:8010/chat"
def invoke_model(model, chat_messages):
    messages=[
                {"role": m["role"], "content": m["content"]}
                for m in chat_messages
            ]

    data = {"model": model, "messages": messages}
    #print('data: \n', data)
    
    res=requests.post("http://localhost:8010/chat", json=data, stream=True)
    return res
    
st.title("ShopTalk Chatbot")

# create sidebar to adjust parameters
st.sidebar.header('Parameters')
temperature = st.sidebar.slider('Temperature', 0.1, 1.0, 0.5, 0.1)
max_tokens = st.sidebar.slider('Max Tokens', 10, 2048, 256, 10)

model_option = st.sidebar.selectbox(
     'Select Model:',
     ('Blip-2', 'llama 3.2'))

st.sidebar.write('You selected:', model_option)


# Set OpenAI API key from Streamlit secrets
#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
if model_option not in st.session_state:
    st.session_state["llm_model"] =  model_option

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        #stream = client.chat.completions.create(
        #    model=st.session_state["openai_model"],
        #    messages=[
        #        {"role": m["role"], "content": m["content"]}
        #        for m in st.session_state.messages
        #    ],
        #    stream=True,
        #)
        #stream = "Same as: " + prompt
        stream = invoke_model(st.session_state["llm_model"], st.session_state.messages)
        # st.write_stream(stream)  # Display stream in Streamlit app
        response = stream.json()  # remove this once the model is plugged in
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response })
        
    
c3 = st.container(border=True)
with c3:
    user_feedback=st.text_input("Enter your feedback of AI Bot Response: ")
    st.button("Submit")
    #st.write("Thank you. Your feedback is shared with our SMEs for review.")

                