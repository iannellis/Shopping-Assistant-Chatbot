from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from util import Chroma_Collection_Connection, connect_ollama_llm, ABO_Dataset

from typing import Iterator

class Agent():
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
    def __init__(self):
        self.chroma_collection = Chroma_Collection_Connection()
        self.llm = connect_ollama_llm()
        self.abo_dataset = ABO_Dataset()
        self.graph = self._build_graph()
    
    def prompt(self, chat_id: str, prompt: str="", image_b64: str=None) -> Iterator:
        config = {"configurable": {"thread_id": chat_id}}

        input_message = {
            "role": "user",
            "content": prompt,
        }

        if image_b64:
            input_message["image_b64"] = image_b64

        return self.graph.stream({"messages": [input_message]}, stream_mode="messages", config=config)
    
    @tool(response_format="content_and_artifact")
    def _retrieve_products(self, query: str | dict):
        """Call with a query about a product a user might shop for to get possible
        matches and details about those matches. Always call if an image is provided.
        In any other circumstance, call with an empty string: ''."""
        if isinstance(query, str):
            if not query:
                # dirty hack
                return '', []
            match_ids = self.chroma_collection.query_image_text(image_b64=None, text=query)
        elif isinstance(query, dict):
            image_b64 = query['image_b64']
            text = query['text']
            match_ids = self.chroma_collection.query_image_text(image_b64=image_b64, text=text)   
            
        if not match_ids:
            return 'Nothing found.', [] 
        image_item_pairs_data = self.abo_dataset.get_image_item_pairs_data(match_ids)
        
        images_b64 = []
        product_data = []
        for i, pair in enumerate(image_item_pairs_data):
            images_b64.append(pair.image_b64)
            product_data.append('Item ' + str(i) + ': ' + pair.item_str)
        return '\n'.join(product_data), images_b64
            
    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    def _query_or_respond(self, state: MessagesState):
        """Generate tool call for retrieval or respond."""
        messages = state['messages']
        last_message = messages[-1]

        # Grab the encoded image. Remove from additional_kwargs to stop confusing
        # LangGraph down the line.
        image_b64 = None
        if "image_b64" in last_message.additional_kwargs:
            image_b64 = last_message.additional_kwargs.pop("image_b64")
            
        # Provide system prompt, bind tool, and invoke model
        if image_b64:
            primary_system_prompt = ChatPromptTemplate([("system", (
                'You are a helpful shopping assistant. The user is providing information '
                'about a product they are shopping for. Build a concise but informative '
                'summary of that product from the user-provided information. Use '
                'conversation history for additional context to help build the summary '
                'if available. Call the "_retrieve_products" tool with the summary. If '
                'the user has not provided any information about a product or no '
                'information at all, call the "_retrieve_products" tool with an empty '
                'string: "".'
                )),
                ("user", "{input}")])
        else:
            primary_system_prompt = ChatPromptTemplate([("system", (
                'You are a helpful shopping assistant. If the user prompt mentions a '
                'product a user might shop for without any other context, the user '
                'mentions they\'re shopping for a product, or in any way implies that '
                'they might want or need a product that they mention, produce a concise '
                'but informative summary of what the user is looking for and use it to '
                'call the "_retrieve_products" tool. Use conversation history for '
                'additional context to help build the summary, if available. If the user '
                'has only provided a broad category of products (such as shoes, but not '
                'carrots), immediately and concisely prompt the user for additional '
                'information without calling the "_retrieve_products" tool.'
                '\n'
                'If the user provides a prompt that does not meet any of the parameters '
                'set out above, call the "_retrieve_products" tool with an empty string: '
                '"".'
                )),
                ("user", "{input}")])
            
        llm_with_tools = primary_system_prompt | self.llm.bind_tools([self._retrieve_products])
        response = llm_with_tools.invoke(state["messages"])
        
        # hack in image handling
        if image_b64 and response.tool_calls:
            original_query = response.tool_calls[0]['args']['query']
            response.tool_calls[0]['args']['query'] = {'text': original_query, 'image_b64': image_b64}
        
        return {"messages": [response]}



    # Step 3: Generate a response using the retrieved content.
    def _generate(self, state: MessagesState):
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
            'You are being provided an enumerated list of products related to the prompt '
            'the user provided. Briefly tell the user that you found some information '
            'related to their query. Then, enumerating each product, provide its name '
            'and a one-sentence summary. For example:\n'
            '   1. The best ketchup in the world\n'
            '      Use this to make that bland food taste great!\n'
            '\n'
            '   2. World’s biggest pumpkin\n'
            '      Make your Halloween awesome.\n'
            '\n'
            'Here is the product information:\n'
            f'{docs_content}'
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = self.llm.invoke(prompt)
        return {"messages": [response]}

    def _build_graph(self):
        # Step 2: Execute the retrieval.
        tools = ToolNode([self._retrieve_products])
        
        # Compile application and test
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node(self._query_or_respond)
        graph_builder.add_node(tools)
        graph_builder.add_node(self._generate)

        graph_builder.set_entry_point("_query_or_respond")
        graph_builder.add_conditional_edges(
            "_query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "_generate")
        graph_builder.add_edge("_generate", END)

        memory = MemorySaver()
        return graph_builder.compile(checkpointer=memory)
    