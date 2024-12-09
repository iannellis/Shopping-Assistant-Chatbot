## Install streamlit using the following command:
## pip install streamlit
## Run the following command to start the app:
## streamlit run app.py

import streamlit as st
#from openai import OpenAI
import json
import os


def get_response(prompt, temperature, max_tokens):
    ## get the API key
    #api_key = os.getenv('OPENAI_API_KEY')
    ## create the client
    #client = OpenAI(api_key=api_key)
    ## create the completion
    #completion = client.chat
    ## get the response
    #response = completion.create(
    #    model='gpt-3.5-turbo',
    #    messages=[
    #        {'role': 'system', 'content': 'You are a helpful assistant.'},
    #        {'role': 'user', 'content': prompt}
    #    ],
    #    temperature=temperature,
    #    max_tokens=max_tokens
    #)
    # return the response
    return prompt




st.title('ShopTalk Chatbot')

# Init the session variables
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    
# create sidebar to adjust parameters
st.sidebar.header('Parameters')
temperature = st.sidebar.slider('Temperature', 0.1, 1.0, 0.5, 0.1)
max_tokens = st.sidebar.slider('Max Tokens', 10, 2048, 256, 10)

# update the interface with the previous messages
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])

# create the interface
if prompt := st.chat_input('What is up?'):
    st.session_state['messages'].append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
        
    # get the response
    response = get_response(prompt, temperature, max_tokens)
    st.session_state['messages'].append({'role': 'assistant', 'content': response})
    with st.chat_message('assistant'):
        st.markdown(response)
    
    
