from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama

import chromadb

import pandas as pd

from httpx import ConnectError
from time import sleep
import os

class Agent():
    def __init__(self):
        self._init_ollama()
        self._init_chroma()
        
    def _init_ollama(self):
        """Setup Ollama connection. Also makes sure model's loaded into video RAM."""
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
        self.llm = llm

    def _init_chroma(self):
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
        
        self.chroma_collection = collection
        self.max_return_items = max_items
        self.chroma_n_return = self.n_return
        

    # load ABO dataset metadata
    abo_fname = "abo-listings-final-draft.pkl"
    abo_dir = os.environ["ABO_DIR_CONTAINER"]
    abo_meta_df = pd.read_pickle(abo_dir + '/' + abo_fname)


    @tool(response_format="content_and_artifact")
    def retrieve(query: str | dict):
        """Call with any topic of any question to retrieve up-to-date information or call
        with an image to get content related to that image."""
        if isinstance(query, str):
            retrieved_docs = retriever.invoke(query)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, []
        elif isinstance(query, dict):
            image = query['image_b64']
            text = query['text']
            display(Image.open(io.BytesIO(base64.b64decode(image))))
            return "The answer is 42.", []
            
    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        messages = state['messages']
        last_message = messages[-1]

        image_b64 = None
        if "image_b64" in last_message.additional_kwargs:
            image_b64 = last_message.additional_kwargs.pop("image_b64")
            
        # Original text-based logic
        primary_system_prompt = ChatPromptTemplate([("system", (
            'You are a helpful question answering assistant. Every time you are queried '
            'about a topic, use the "retrieve" function to gather information that you '
            'will use in the response.')),
            ("user", "{input}")])
        llm_with_tools = primary_system_prompt | llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        
        if image_b64:
            original_query = response.tool_calls[0]['args']['query']
            response.tool_calls[0]['args']['query'] = {'text': original_query, 'image_b64': image_b64}
        
        return {"messages": [response]}

    # Step 2: Execute the retrieval.
    tools = ToolNode([retrieve])

    # Step 3: Generate a response using the retrieved content.
    def generate(state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = llm.invoke(prompt)
        return {"messages": [response]}


    # Compile application and test
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "abc123"}}

    with open('../../assets/sofa.jpg', 'rb') as f:
        image_bytes = f.read()
        
    image_encoded = base64.b64encode(image_bytes)

    input_message = {
        "role": "user",
        "content": "What is Task Decomposition?",
        "image_b64": image_encoded
    }

    for message, metadata in graph.stream({"messages": [input_message]}, stream_mode="messages", config=config):
        if metadata['langgraph_node']=='tools':
            print(message.artifact)
        if metadata['langgraph_node']=='generate':
            print(message.content, end="", flush=True)