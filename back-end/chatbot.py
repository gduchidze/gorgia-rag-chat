import json
import os
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from pinecone import Pinecone
from openai import OpenAI


OPENAI_API_KEY="sk-MckIJPkrp42Ev4_EBkj6aQ"
OPENAI_BASE_URL="https://api.ailab.ge"
MODEL_NAME="tbilisi-ai-lab-2.0"
EMBEDDINGS_API_KEY="hHnQ6vbPMRKj7eCs5IG6QmjFTyYjVccW"
EMBEDDINGS_API_BASE="https://api.deepinfra.com/v1/openai"
EMBEDDINGS_MODEL="BAAI/bge-m3"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
pc = Pinecone(api_key=PINECONE_API_KEY)

class CustomEmbeddings(OpenAIEmbeddings):
    def __init__(self):
        super().__init__(openai_api_key=EMBEDDINGS_API_KEY)
        self.client = OpenAI(api_key=EMBEDDINGS_API_KEY, base_url=EMBEDDINGS_API_BASE)
        self.model = EMBEDDINGS_MODEL

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding


@tool("search_products")
def search_products(query: str) -> str:
    """Search for products in Gorgia database."""
    try:
        llm = ChatOpenAI(
            model_name=MODEL_NAME,
            openai_api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0.1
        )

        clean_query = rewrite_query(llm, query)
        print(f"Clean query: {clean_query}")

        vector_store = PineconeVectorStore(
            embedding=CustomEmbeddings(),
            index_name="gorgia",
            namespace="gorgia_products"
        )
        results = vector_store.similarity_search(clean_query, k=10)

        if not results:
            return json.dumps({
                "type": "product_list",
                "message": "სამწუხაროდ აღნიშნული პროდუქტი ვერ მოიძებნა.",
                "products": []
            }, ensure_ascii=False)

        products = []
        for doc in results:
            image_url = doc.metadata.get('image_url', '')
            if image_url:
                image_url = f"https://images.weserv.nl/?url={image_url}&w=400&h=400&fit=contain&bg=white&q=90"

            product = {
                "name": doc.metadata.get('name', 'N/A'),
                "category": doc.metadata.get('category', 'N/A'),
                "price": doc.metadata.get('price', 'N/A'),
                "url": doc.metadata.get('product_url', '#'),
                "image_url": image_url or 'https://placehold.co/400x400?text=No+Image',
                "description": doc.metadata.get('description', '')
            }
            products.append(product)

        return json.dumps({
            "type": "product_list",
            "message": f"თქვენს მიერ მოთხოვნილი პროდუქტები:",
            "products": products
        }, ensure_ascii=False)

    except Exception as e:
        print(f"Error in search_products: {str(e)}")
        return json.dumps({
            "type": "error",
            "message": "სამწუხაროდ, პროდუქტების ძიებისას დაფიქსირდა შეცდომა.",
            "products": []
        }, ensure_ascii=False)


@tool("search_docs")
def search_docs(query: str) -> str:
    """Search for all kind of general information."""
    try:
        llm = ChatOpenAI(
            model_name=MODEL_NAME,
            openai_api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0
        )

        clean_query = rewrite_query(llm, query)

        vector_store = PineconeVectorStore(
            embedding=CustomEmbeddings(),
            index_name="gorgia",
            namespace="gorgia_docs"
        )
        results = vector_store.similarity_search(clean_query, k=1)
        return "\n".join([f"{i}:\n{doc.page_content}" for i, doc in enumerate(results)])
    except Exception as e:
        return f"დოკუმენტის ძიების შეცდომა: {str(e)}"


def rewrite_query(llm, user_input: str) -> str:
    """
    Rewrites user input into a clean search query.
    """
    prompt = """
    გადააკეთე მომხმარებლის შეკითხვა მხოლოდ საძიებო ტექსტად.
    დატოვე მხოლოდ საძიებო სიტყვები.

    შეკითხვა: "{input}"
    ახალი შეკითხვა: """

    response = llm.invoke([
        SystemMessage(content=prompt.format(input=user_input))
    ])
    return response.content.strip()


class GorgiaAgent:
    def __init__(self, llm, tools, system_prompt):
        self.llm = llm
        self.tools = tools
        self.chat_history = []

        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        self.system_prompt = f"""
        {system_prompt}

        არსებული ინსტრუმენტები:
        {tool_descriptions}
"""

    def _get_tool_decision(self, query: str) -> str:
            decision_prompt = """
    შეაფასე მომხმარებლის შეკითხვა და გადაწყვიტე რომელი ინსტრუმენტი უნდა გამოვიყენოთ.

    წესები:

    1. "search_docs" გამოიყენე ყველა სახის ზოგადი შეკითხვისთვის

    2. "search_products" გამოიყენე მხოლოდ როცა: ნებისმიერი სახით პროდუქტის მოთხოვნაზე,კონკრეტულ პროდუქტზე, ბრენდზე ან პროდუქტის ფასზე

    3. "none" გამოიყენე მხოლოდ:
       - მისალმება/დამშვიდობება
       - მარტივი მადლობა

    აუცილებლად გამოიყენე search_docs თუ კითხვა შეიცავს ინფორმაციის მოთხოვნას პროცედურების, წესების ან სერვისების შესახებ.

    მიმდინარე შეკითხვა: "{query}"

    დააბრუნე მხოლოდ ერთი სიტყვა: search_docs, search_products ან none.
    პასუხი:"""

            decision = self.llm.invoke([
                SystemMessage(content=decision_prompt),
                HumanMessage(content=query)
            ]).content.strip().lower()

            return decision

    def _execute_tool(self, tool_name: str, tool_input: str) -> dict:
        """Execute a tool and return its response"""
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            return {"type": "error", "message": "Tool not found"}

        try:
            response = tool.invoke(tool_input)

            if tool_name == "search_products":
                try:
                    return json.loads(response)
                except:
                    return {"type": "error", "message": "Failed to parse products response"}

            elif tool_name == "search_docs":
                format_prompt = """
    მოცემულია მომხმარებლის კითხვა და საინფორმაციო ბაზიდან მოძიებული ინფორმაცია.
    გთხოვთ, დააფორმატოთ პასუხი ისე, რომ:
    1. იყოს პირდაპირი პასუხი კითხვაზე
    2. იყოს მარტივად გასაგები
    3. შეინარჩუნოს ყველა მნიშვნელოვანი დეტალი
    4. საჭიროების შემთხვევაში დაალაგოთ ნაბიჯები

    კითხვა: {tool_input}

    მოძიებული ინფორმაცია: {response}

    გთხოვთ, დააფორმატოთ პასუხი ისე, რომ პირდაპირ უპასუხოთ მომხმარებლის კითხვას."""

                formatted = self.llm.invoke([
                    SystemMessage(content=format_prompt.format(
                        tool_input=tool_input,
                        response=response
                    ))
                ]).content

                return {
                    "type": "text",
                    "message": formatted
                }

        except Exception as e:
            return {"type": "error", "message": str(e)}

    def run(self, user_input: str) -> dict:
        tool_to_use = self._get_tool_decision(user_input)
        print(f"Selected tool: {tool_to_use}")
        self.chat_history.append(HumanMessage(content=user_input))

        if tool_to_use == "none":
            messages = [
                SystemMessage(content=self.system_prompt + "სიტემის და მომხმარებლის ჩატის ისტორია: \n".join([str(m) for m in self.chat_history])),
                *self.chat_history
            ]
            response = self.llm.invoke(messages)
            self.chat_history.append(SystemMessage(content=response.content))

            return {
                "type": "text",
                "message": response.content
            }

        try:
            response = self._execute_tool(tool_to_use, user_input)

            if isinstance(response, dict):
                if response.get("type") == "product_list":
                    history_content = f"Found {len(response.get('products', []))} products: " + \
                                      ", ".join([p.get('name', 'N/A') for p in response.get('products', [])])
                else:
                    history_content = response.get("message", str(response))
            else:
                history_content = str(response)

            self.chat_history.append(SystemMessage(content=history_content))

            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]

            print(f"Chat history after: {self.chat_history}")
            return response

        except Exception as e:
            print(f"Error in run method: {str(e)}")
            error_response = {
                "type": "error",
                "message": f"დაფიქსირდა შეცდომა: {str(e)}"
            }
            self.chat_history.append(SystemMessage(content=f"Error: {str(e)}"))
            return error_response

    def clear_history(self):
        self.chat_history = []

def create_gorgia_agent():
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.1
    )
    tools = [search_products, search_docs]
    system_prompt = """
    GORGIA არის უმსხვილესი კორპორაცია სამშენებლო და სარემონტო სფეროში კავკასიაში.    
    თქვენ ხართ Gorgia-ს დახმარების ასისტენტი. თქვენი მიზანია მომხმარებლებს მიაწოდოთ ზუსტი და სასარგებლო ინფორმაცია ბუნებრივი დიალოგის ფორმით.

    ძირითადი წესები:
    1. ყოველთვის გამოიყენეთ search_docs ზოგადი ინფორმაციისთვის რომ არ მიაწოდო არასწორი ინფორმაცია მომხმარებელს ყოველთვის შეამოწმე სპეციალურ დოკუმენტში.
    2. გამოიყენეთ search_products პროდუქტების ან ბრენდების შესახებ ინფორმაციისთვის
    3. პასუხები უნდა იყოს მოკლე, გასაგები და ინფორმატიული
    4. არ ახსენოთ ინსტრუმენტები ან ტექნიკური დეტალები პასუხებში
    
    როცა მოგეწოდებათ ინფორმაცია, გთხოვთ:
    1. გადააფორმატეთ ის მარტივ, გასაგებ პუნქტებად
    2. მოაშორეთ ტექნიკური დეტალები და ფორმატირება
    3. დაწერეთ ბუნებრივი, საუბრის სტილში

    პასუხის ფორმატი:
    - გამოიყენეთ მარტივი, გასაგები ენა
    - დაასრულეთ დამატებითი დახმარების შეთავაზებით, თუ საჭიროა
    """
    return GorgiaAgent(llm, tools, system_prompt)
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
