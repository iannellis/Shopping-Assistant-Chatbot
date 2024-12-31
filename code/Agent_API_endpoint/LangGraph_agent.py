from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from util import Chroma_Collection_Connection, connect_ollama_llm, ABO_Dataset

from icecream import ic

"""The LangGraph agent, which takes a user's query, retrieves relevant product
information, and uses that information to respond to a user's queries.

Note that there are a couple dirty hacks involved to get LangGraph to behave the way
we want it to. First, we have hacked in the ability to pass a user's submitted image
through the graph. Second, we've tried to force the LLM to always call the tool, and
if it doesn't want to, it just calls it with an empty string. The reason is that 
during 'query_or_respond', the LLM is called and decides whether to call the tool or
to respond to the user immediately. Because thhe LLM was called with the possiblity
of a tool call response, it is called with the 'tools' keyword argument, which
disables streaming. Therefore, any response the LLM decides to provide immediately
is not streamed. The solution is to make the LLM call the tool with nothing (to
which the tool responds with nothing), then respond to the user during the next
call. The two calls have lower latency than waiting for the LLM to completely
generate what could be a long response."""

chroma_collection = Chroma_Collection_Connection()
llm = connect_ollama_llm()
abo_dataset = ABO_Dataset()

# store the images users upload for future recall in the UI
user_images = dict()
    
@tool(response_format="content_and_artifact")
def retrieve_products(query: str | dict):
    """Call with a query about a product a user might shop for to get possible
    matches and details about those matches. In any other circumstance, call with an 
    empty string: ''."""
    ic("In retrieve_products")
    if isinstance(query, str):
        ic(query)
        if not query:
            # streaming dirty hack
            return '', []
        match_ids = chroma_collection.query_image_text(image_b64=None, text=query)
    elif isinstance(query, dict):
        ic(query["text"])
        image_b64 = query['image_b64']
        text = query['text']
        match_ids = chroma_collection.query_image_text(image_b64=image_b64, text=text)   
        # ic(match_ids)
    if not match_ids:
        return 'Nothing found.', [] 
    image_item_pairs_data = abo_dataset.get_image_item_pairs_data(match_ids)
    
    # ic(image_item_pairs_data)
    images_b64 = []
    product_data = []
    for i, pair in enumerate(image_item_pairs_data):
        images_b64.append(pair.image_b64)
        product_data.append('Item ' + str(i+1) + ': ' + pair.item_str)
    return '\n\n'.join(product_data), images_b64
            
# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    messages = state['messages']
    last_message = messages[-1]

    # Grab the encoded image. Remove from additional_kwargs to stop confusing
    # LangGraph down the line.
    image_b64 = None
    if "image_b64" in last_message.additional_kwargs:
        image_b64 = last_message.additional_kwargs.pop("image_b64")
        
    # Provide system prompt, bind tool, and invoke model
    # ic('In query_or_respond')
    if image_b64:
        primary_system_prompt = ChatPromptTemplate([("system", (
            'You are a helpful shopping assistant. The user is providing information '
            'about a product they are shopping for. Build a concise but informative '
            'summary of that product from the user-provided information. Use '
            'conversation history for additional context to help build the summary '
            'if available. Call the "retrieve_products" tool with the summary. If '
            'the user has not provided any information about a product or no '
            'information at all, call the "retrieve_products" tool with an empty '
            'string: "".'
            )),
            ("user", "{input}")])
    else:
        primary_system_prompt = ChatPromptTemplate([("system", (
            'You are a helpful shopping assistant. If the user prompt mentions '
            'a product a user might shop for without any other context, the user '
            'mentions they\'re shopping for a product, or in any way implies that '
            'they might want or need a product that they mention, produce a concise '
            'but informative summary of what the user is looking for and use it to '
            'call the "retrieve_products" tool. Use conversation history for '
            'additional context to help build the summary, if available. If the user '
            'has only provided a broad category of products (such as shoes, but not '
            'carrots), immediately and concisely prompt the user for additional '
            'information without calling the "retrieve_products" tool.'
            '\n'
            'If the user provides a prompt that does not meet any of the parameters '
            'set out above, call the "retrieve_products" tool with an empty string: '
            '"".'
            )),
            ("user", "{input}")])
        
    # ic(primary_system_prompt)
    llm_with_tools = primary_system_prompt | llm.bind_tools([retrieve_products])
    response = llm_with_tools.invoke(state["messages"])
    # ic(response)
    # hack in image handling
    if image_b64 and response.tool_calls:
        original_query = response.tool_calls[0]['args']['query']
        response.tool_calls[0]['args']['query'] = {'text': original_query, 'image_b64': image_b64}
    
    return {"messages": [response]}

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

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    # ic(tool_messages)
    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    ic(docs_content)
    if docs_content:
        system_message_content = (
            'You are being provided an enumerated list of products. Briefly tell the '
            'user that you found the closest matches to what they\'re looking for in '
            'your database. Then, using only the information provided below, enumerate '
            'each product, provide its name, then a one or two-sentence summary. For '
            'example:\n'
            '   1. **The best ketchup in the world**\n'
            '      Rated #1 by the World Ketchup Forum, this ketchup will jazz up your '
            '      food so that your guests never complain again.\n'
            '\n'
            '   2. **World’s biggest pumpkin**\n'
            '      Making it into the Gueniss Book of World Records, this pumpkin weighs '
            '      in at one ton. With its stunning orange color, it\'s a must for '
            '      anybody aiming to throw the biggest Halloween party  in town.\n'
            '\n'
            'Summarize any information provided about each product, even if it does not '
            'match the user prompt. '
            'Always list all the items and provide a summary of each. '
            'List the items in the order they were provided to you. '
            'Do not elaborate beyond using information provided below. Here is the '
            'product information:\n'
            f'{docs_content}'
        )
        prompt = [SystemMessage(system_message_content)] + conversation_messages
    else: # more streaming dirty hack
        prompt = conversation_messages

    # Run
    # ic(prompt)
    response = llm.invoke(prompt)
    # ic(response)
    return {"messages": [response]}

# Step 2: Execute the retrieval.
tools = ToolNode([retrieve_products])

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
    
# For outside funtion to call
async def prompt(chat_id: str, prompt: str="", image_b64: str=None):
    """Run a user's prompt through the model"""
    config = {"configurable": {"thread_id": chat_id}}

    input_message = {"role": "user", "content": prompt}
    if image_b64:
        input_message["image_b64"] = image_b64
        user_images[chat_id] = image_b64

    for message, metadata in graph.stream({"messages": [input_message]}, stream_mode="messages", config=config):
        if metadata['langgraph_node']=='tools' and message.artifact:
            yield {"images": message.artifact, "text": ""}
        if metadata['langgraph_node']=='generate':
            yield {"images": [], "text": message.content}
            
def get_checkpoint_ids():
    """Get a list of all memory checkpoints."""
    checkpoints = memory.list()
    return checkpoints

def retrieve_checkpoint(chat_id: str):
    """Retrieve a chat thread."""
    config = {"configurable": {"thread_id": chat_id}}
    checkpoint = memory.list(config)
    return checkpoint, user_images[chat_id]