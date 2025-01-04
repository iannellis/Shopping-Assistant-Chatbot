from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from icecream import ic

from util import Chroma_Collection_Connection, connect_ollama_llm, ABO_Dataset

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
    
@tool(response_format="content_and_artifact")
def retrieve_products(query: str | dict):
    """Call with a query about a product a user might shop for to get possible
    matches and details about those matches. The query may be up to a sentence long. 
    In any other circumstance, call with an empty string: ''."""
    # ic("In retrieve_products")
    if isinstance(query, str):
        # ic(query)
        if not query:
            # streaming dirty hack
            return '', []
        match_item_ids = chroma_collection.query_text(text=query)
    elif isinstance(query, dict):
        # ic(query["text"])
        match_item_ids = chroma_collection.query_image_text(**query)   
        # ic(match_item_ids)
    else:
        raise Exception('Invalid query type')
    
    
    if not match_item_ids:
        return 'No matching products found.', [] 
    image_item_pairs_data = abo_dataset.get_items_data(match_item_ids)
    
    # ic(image_item_pairs_data)
    images_b64 = []
    product_data = []
    for i, pair in enumerate(image_item_pairs_data):
        images_b64.append(pair.image_b64)
        product_data.append(str(i+1) + ': (' + pair.item_str + ')')
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
        primary_system_prompt = (
            'You are a helpful shopping assistant. The user is providing information '
            'about a product they are shopping for. Build a concise but informative '
            'summary of that product from the user-provided information. Use '
            'conversation history for additional context to help build the summary, '
            'if available. Call the "retrieve_products" tool with the summary. If '
            'the user has not provided any information about a product or no '
            'information at all, call the "retrieve_products" tool with an empty '
            'string: "".'
            )
    else:
        primary_system_prompt = (
            'You are a helpful shopping assistant. If the user prompt mentions '
            'a product a user might shop for without any other context, the user '
            'mentions they\'re shopping for a product, or in any way implies that '
            'they might want or need a product that they mention, produce a concise '
            'but informative summary of what the user is looking for and use it to '
            'call the "retrieve_products" tool. Use conversation history for '
            'additional context to help build the summary, if available. If the user '
            'has only provided a broad category of products (such as shoes, but not '
            'carrots), immediately and concisely prompt the user for additional '
            'information without calling the "retrieve_products" tool. '
            'If the user provides a prompt that does not meet any of the parameters '
            'set out above, call the "retrieve_products" tool with an empty string: '
            '"".'
            )
   
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    prompt = [SystemMessage(primary_system_prompt)] + conversation_messages
    # ic(prompt)
    llm_with_tools = llm.bind_tools([retrieve_products])
    response = llm_with_tools.invoke(prompt)
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
    # ic(conversation_messages)
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    # ic(docs_content)
    if docs_content:
        system_message_content = (
            'You are being provided an enumerated list of products. Briefly tell the '
            'user that you found the closest matches to what they\'re looking for in '
            'your database. Then, using only the information provided below, enumerate '
            'each product, provide its name, then a 1 or 2-sentence summary. For '
            'example:\n'
            '   1. **The best ketchup in the world**\n'
            '      Rated #1 by the World Ketchup Forum, this ketchup will jazz up your '
            '      food so that your guests never complain about its blandness again.\n'
            '\n'
            '   2. **World’s biggest pumpkin**\n'
            '      Making it into the Gueniss Book of World Records, this pumpkin weighs '
            '      in at one ton. With its stunning orange color, it\'s a must for '
            '      anybody aiming to throw the biggest Halloween party in town.\n'
            '\n'
            '   3. **Military-grade water bottle**\n'
            '      If your water bottle is always getting broken due to your strenuous '
            '      activities, this is the water bottle for you. Tested to withstand up '
            '      to one ton of weight.\n'
            'Summarize any information provided about each product, even if it does not '
            'match the user prompt. '
            'Always list all the items and provide a summary of each. '
            'Only use the information below in this system prompt to develop the summaries. '
            'List the items in the order they were provided to you. Here is the '
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
    
# For outside funtion to calls
async def prompt(thread_id: str, prompt_str: str="", image_b64: str=""):
    """Run a user's prompt through the model"""
    config = {"configurable": {"thread_id": thread_id}}

    input_message = {"role": "user", "content": prompt_str}
    if image_b64:
        input_message["image_b64"] = image_b64

    for message, metadata in graph.stream({"messages": [input_message]}, stream_mode="messages", config=config):
        if metadata['langgraph_node']=='tools' and message.artifact:
            yield {"images": message.artifact, "text": ""}
        if metadata['langgraph_node']=='generate':
            yield {"images": [], "text": message.content}
            
def get_thread_ids():
    """Get a list of all memory checkpoints."""
    return list(memory.storage.keys())

def get_message_thread(thread_id: str):
    """Retrieve a chat thread."""
    config = {"configurable": {"thread_id": thread_id}}
    checkpoints = list(reversed(list(memory.list(config))))
    conversation = []
    user_image = None
    i = 0
    while i < len(checkpoints):
        data = checkpoints[i][2]['writes']
        if not data:
            i += 1
            continue
        
        if '__start__' in data:
            content = data['__start__']['messages'][0]['content']
            conversation.append({'role': 'user', 'images': [], 'content': content})
            if 'image_b64' in data['__start__']['messages'][0]:
                user_image = data['__start__']['messages'][0]['image_b64']
            i += 1
        # I tried to instruct the LLM to avoid outputting from here, but it might happen
        elif 'query_or_respond' in data:
            if data['query_or_respond']['messages'][0].content:
                content = data['query_or_respond']['messages'][0].content
                conversation.append({'role': 'ai', 'images': [], 'content': content})
            i += 1
        # 'generate' output always follows a tool call and the reverse is true, so do them together
        elif 'tools' in data:
            images = data['tools']['messages'][0].artifact
            data = checkpoints[i+1][2]['writes']
            content = data['generate']['messages'][0].content
            conversation.append({'role': 'ai', 'images': images, 'content': content})
            i += 2
        else:
            print('Problem in get_message_thread')
    
    return conversation, user_image