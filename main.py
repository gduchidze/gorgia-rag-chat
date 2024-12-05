# import os
# from typing import List
# from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_core.tools import tool
# from pinecone import Pinecone
# from openai import OpenAI
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
# @tool("search_products")
# def search_products(query: str) -> str:
#     """Search for products in Gorgia database."""
#     try:
#         vector_store = PineconeVectorStore(
#             embedding=CustomEmbeddings(),
#             index_name="gorgia",
#             namespace="gorgia_products"
#         )
#         results = vector_store.similarity_search(query, k=10)
#         if not results:
#             return "სამწუხაროდ აღნიშნული პროდუქტი ვერ მოიძებნა."
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
# @tool("search_docs")
# def search_docs(query: str) -> str:
#     """Search any kind of general information about Gorgia."""
#     try:
#         vector_store = PineconeVectorStore(
#             embedding=CustomEmbeddings(),
#             index_name="gorgia",
#             namespace="gorgia_docs"
#         )
#         results = vector_store.similarity_search(query, k=1)
#         return "\n".join([f"{i}:\n{doc.page_content}" for i, doc in enumerate(results)])
#     except Exception as e:
#         return f"დოკუმენტის ძიების შეცდომა: {str(e)}"
#
#
# class GorgiaAgent:
#     def __init__(self, llm, tools, system_prompt):
#         self.llm = llm
#         self.system_prompt = system_prompt
#         self.tools = tools
#
#     def run(self, user_input: str, chat_history: List[AnyMessage] = None) -> str:
#         chat_history = chat_history or []
#         tools_description = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
#
#         messages = [
#             SystemMessage(content=f"{self.system_prompt}\n\nAvailable tools:\n{tools_description}\n\n"
#                                   "To use a tool, respond with: <tool>tool_name|input</tool>. "
#                                   "If no tool is needed, respond directly.")
#         ]
#
#         messages.extend(chat_history)
#         messages.append(HumanMessage(content=user_input))
#         response = self.llm.invoke(messages)
#
#         # Check if the response contains a tool invocation
#         if "<tool>" in response.content and "</tool>" in response.content:
#             tool_response = self._handle_tool_invocation(response.content.strip("<tool>").strip("</tool>"))
#             if tool_response:
#                 if "search_docs" in tool_response:
#                     llm_response = self.llm.invoke(
#                         messages + [SystemMessage(content=tool_response)]
#                     )
#                     return llm_response.content
#                 else:
#                     return tool_response
#
#         return response.content
#
#     def _handle_tool_invocation(self, tool_message: str) -> str:
#         try:
#             tool_name, tool_input = tool_message.split("|", 1)
#             tool = next((t for t in self.tools if t.name == tool_name), None)
#             if tool:
#                 result = tool.invoke(tool_input)
#                 return self._generate_tool_response(result, tool_name)
#             else:
#                 return f"Tool '{tool_name}' not found."
#         except ValueError:
#             return "Invalid tool response format. Please ensure it follows '<tool>tool_name|input</tool>'."
#         except Exception as e:
#             return f"Error invoking tool: {str(e)}"
#
#     def _generate_tool_response(self, tool_result: str, tool_name: str) -> str:
#         """Generate the final response after tool invocation."""
#         tool_message = f"Tool '{tool_name}' returned: {tool_result}"
#         return tool_message
#
# def create_gorgia_agent():
#     llm = ChatOpenAI(
#         model=MODEL_NAME,
#         openai_api_key=OPENAI_API_KEY,
#         base_url=OPENAI_BASE_URL,
#         temperature=0.1
#     )
#     tools = [search_products, search_docs]
#     system_prompt = """
#     თქვენ ხართ Gorgia კომპანიის დახმარების ასისტენტი.
#     გამოიყენეთ მოცემული ინსტრუმენტები საჭიროებისამებრ:
#     - პროდუქტის შესახებ informacijisთვის გამოიყენეთ search_products
#     - ზოგადი ინფორმაციისთვის გამოიყენეთ search_docs
#
#     ფორმატი: <tool>tool_name|input</tool>
#     პასუხები დაწერეთ ქართულად. იყავით თავაზიანი და მოკლე.
#     """
#     return GorgiaAgent(llm, tools, system_prompt)
#
# if __name__ == "__main__":
#     agent = create_gorgia_agent()
#     chat_history = []
#     print("Gorgia ჩატბოტი დაიწყო! დასასრულებლად აკრიფეთ 'exit' ან 'quit'")
#     while True:
#         try:
#             user_input = input("\nთქვენი კითხვა: ")
#             if user_input.lower() in ['exit', 'quit']:
#                 print("\nმადლობა რომ ისარგებლეთ Gorgia ჩატბოტით!")
#                 break
#             print("\nვეძებ პასუხს...")
#             response = agent.run(user_input, chat_history)
#             print("\nპასუხი:")
#             print(response)
#             chat_history.extend([HumanMessage(content=user_input), SystemMessage(content=response)])
#             print(chat_history)
#         except Exception as e:
#             print(f"\nშეცდომა: {str(e)}")
