# import json
# import operator
# from typing import List, TypedDict, Annotated, Optional, Dict, Any
#
# from langchain_core.agents import AgentAction
# from langchain_core.messages import BaseMessage
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langgraph.graph import StateGraph, END
# from pinecone import Pinecone
# from openai import OpenAI
# import os
# from langchain_core.tools import tool
#
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
# MODEL_NAME = os.getenv("MODEL_NAME")
#
# EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY")
# EMBEDDINGS_API_BASE = os.getenv("EMBEDDINGS_API_BASE")
# EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
#
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
# pc = Pinecone(api_key=PINECONE_API_KEY)
#
#
#
# class AgentState(TypedDict):
#    input: str
#    chat_history: list[BaseMessage]
#    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
#
#
# class CustomEmbeddings(OpenAIEmbeddings):
#     def __init__(self):
#         super().__init__(openai_api_key=EMBEDDINGS_API_KEY)
#         self.client = OpenAI(api_key=EMBEDDINGS_API_KEY, base_url=EMBEDDINGS_API_BASE)
#         self.model = EMBEDDINGS_MODEL
#
#     def embed_query(self, text: str) -> List[float]:
#         response = self.client.embeddings.create(
#             model=self.model,
#             input=text,
#             encoding_format="float"
#         )
#         return response.data[0].embedding
#
#
#
# @tool ("search_products")
# def search_products(query: str) -> str:
#     """Search for products in Gorgia database.
#
#     Args:
#         query: The search query for products
#
#     Returns:
#         Information about matched products
#     """
#     try:
#         vector_store = PineconeVectorStore(
#             embedding=CustomEmbeddings(),
#             index_name="gorgia",
#             namespace="gorgia_products"
#         )
#         results = vector_store.similarity_search(query, k=5)
#         formatted_results = [
#             f"პროდუქტი {i + 1}:\n"
#             f"სახელი: {doc.metadata.get('name', 'N/A')}\n"
#             f"კატეგორია: {doc.metadata.get('category', 'N/A')}\n"
#             f"ფასი: {doc.metadata.get('price', 'N/A')}\n"
#             f"ბმული: {doc.metadata.get('product_url', 'N/A')}\n"
#             for i, doc in enumerate(results)
#         ]
#         return "\n".join(formatted_results) + f"\nმოიძებნა {len(results)} პროდუქტი.\n"
#     except Exception as e:
#         return f"პროდუქტის ძიების შეცდომა: {str(e)}"
#
# @tool ("search_docs")
# def search_docs(query: str) -> str:
#     """Search for information in Gorgia documentation.
#
#     Args:
#         query: The search query for documentation
#
#     Returns:
#         Relevant documentation content
#     """
#     try:
#         vector_store = PineconeVectorStore(
#             embedding=CustomEmbeddings(),
#             index_name="gorgia",
#             namespace="gorgia_docs"
#         )
#         results = vector_store.similarity_search(query, k=3)
#         return "\n".join([f"დოკუმენტი {i + 1}:\n{doc.page_content}" for i, doc in enumerate(results)])
#     except Exception as e:
#         return f"დოკუმენტაციის ძიების შეცდომა: {str(e)}"
#
# @tool("final_answer")
# def final_answer(
#    introduction: str,
#    research_steps: str,
#    main_body: str,
#    conclusion: str,
#    sources: str
# ):
#     """
#     Generate a final formatted answer.
#
#     Args:
#         introduction: The introduction text of the response.
#         research_steps: The steps taken during the research.
#         main_body: The main body of the response.
#         conclusion: The concluding remarks of the response.
#         sources: The sources used for the response.
#
#     Returns:
#         A formatted response combining all the input sections.
#     """
#     if type(research_steps) is list:
#         research_steps = "\n".join([f"- {r}" for r in research_steps])
#     if type(sources) is list:
#         sources = "\n".join([f"- {s}" for s in sources])
#     return ""
#
#
# system_prompt = """You are a helpful assistant for the Gorgia company.
# Your task is to help customers by providing accurate information about products
# and general company information.
#
# You have access to two tools:
# 1. search_products: Use this for any product-related queries
# 2. search_docs: Use this for company information, policies, and general questions
#
# Always provide responses in Georgian unless specifically asked in another language.
# Be polite, professional, and ensure your answers are accurate based on the tool results.
#
# When using search tools, analyze the results carefully and provide comprehensive answers.
# If you don't find specific information, be honest about it and suggest alternatives."""
#
# prompt = ChatPromptTemplate.from_messages([
#    ("system", system_prompt),
#    MessagesPlaceholder(variable_name="chat_history"),
#    ("user", "{input}"),
#    ("assistant", "scratchpad: {scratchpad}"),
# ])
#
# llm = ChatOpenAI(
#     model=MODEL_NAME,
#     openai_api_key=OPENAI_API_KEY,
#     base_url=OPENAI_BASE_URL,
#     temperature=0.1
#
# )
#
# tools = [search_products,search_docs, final_answer]
#
# def create_scratchpad(intermediate_steps: list[AgentAction]):
#    research_steps = []
#    for i, action in enumerate(intermediate_steps):
#        if action.log != "TBD":
#            research_steps.append(
#                f"Tool: {action.tool}, input: {action.tool_input}\n"
#                f"Output: {action.log}"
#            )
#    return "\n---\n".join(research_steps)
#
# oracle = (
#     {
#         "input": lambda x: x["input"],
#         "chat_history": lambda x: x["chat_history"],
#         "scratchpad": lambda x: create_scratchpad(
#             intermediate_steps=x["intermediate_steps"]
#         ),
#     }
#     | prompt
#     | llm.bind_tools(tools, tool_choice="auto")
# )
#
# inputs = {
#     "input": "tell me something interesting about gorgia",
#     "chat_history": [],
#     "intermediate_steps": [],
# }
# out = oracle.invoke(inputs)
# # print(out)
# # print(out.tool_calls[0]["args"])
#
# def run_oracle(state: TypedDict):
#    print("run_oracle")
#    print(f"intermediate_steps: {state['intermediate_steps']}")
#    out = oracle.invoke(state)
#    tool_name = out.tool_calls[0]["name"]
#    tool_args = out.tool_calls[0]["args"]
#    action_out = AgentAction(
#        tool=tool_name,
#        tool_input=tool_args,
#        log="TBD"
#    )
#    return {
#        "intermediate_steps": [action_out]
#    }
#
#
# def router(state: TypedDict):
#    # return the tool name to use
#    if isinstance(state["intermediate_steps"], list):
#        return state["intermediate_steps"][-1].tool
#    else:
#        # if we output bad format go to final answer
#        print("Router invalid format")
#        return "final_answer"
#
# tool_str_to_func = {
#     "search_products": search_products,
#     "search_docs": search_docs,
#    "final_answer": final_answer
# }
#
#
# def run_tool(state: TypedDict):
#    # use this as helper function so we repeat less code
#    tool_name = state["intermediate_steps"][-1].tool
#    tool_args = state["intermediate_steps"][-1].tool_input
#    print(f"{tool_name}.invoke(input={tool_args})")
#    # run tool
#    out = tool_str_to_func[tool_name].invoke(input=tool_args)
#    action_out = AgentAction(
#        tool=tool_name,
#        tool_input=tool_args,
#        log=str(out)
#    )
#    return {"intermediate_steps": [action_out]}
#
# from langgraph.graph import StateGraph, END
#
#
# # initialize the graph with our AgentState
# graph = StateGraph(AgentState)
#
#
# # add nodes
# graph.add_node("oracle", run_oracle)
# graph.add_node("search_products", run_tool)
# graph.add_node("search_docs", run_tool)
# graph.add_node("final_answer", run_tool)
#
#
# # specify the entry node
# graph.set_entry_point("oracle")
#
#
# # add the conditional edges which use the router
# graph.add_conditional_edges(
#    source="oracle",  # where in graph to start
#    path=router,  # function to determine which node is called
# )
#
#
# # create edges from each tool back to the oracle
# for tool_obj in tools:
#    if tool_obj.name != "final_answer":
#        graph.add_edge(tool_obj.name, "oracle")
#
#
# # if anything goes to final answer, it must then move to END
# graph.add_edge("final_answer", END)
#
#
# # finally, we compile our graph
# runnable = graph.compile()
#
# out_1 = runnable.invoke({
#     "input": "tell me something interesting about Gorgia",
#     "chat_history": [],
# })
#
# print(out_1)