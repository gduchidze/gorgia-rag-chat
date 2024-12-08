import logging
import os
from typing import TypedDict, Dict, List
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, END
from langchain_community.embeddings import DeepInfraEmbeddings


OPENAI_API_KEY = "sk-MckIJPkrp42Ev4_EBkj6aQ"
OPENAI_BASE_URL = "https://api.ailab.ge"
MODEL_NAME = "tbilisi-ai-lab-2.0"
DEEPINFRA_API_TOKEN = "hHnQ6vbPMRKj7eCs5IG6QmjFTyYjVccW"
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


class GorgiaState(TypedDict):
    messages: List[BaseMessage]
    next_step: str
    current_query: str
    products: List[dict]
    docs: str
    response: Dict


class CustomEmbeddings(DeepInfraEmbeddings):
    def __init__(self):
        super().__init__(
            model_id="BAAI/bge-m3",
            deepinfra_api_token=DEEPINFRA_API_TOKEN
        )


class SearchProductsTool(BaseTool):
    name: str = "search_products"
    description: str = "Search for products in Gorgia database"
    return_direct: bool = True

    def _run(self, query: str, messages: List[BaseMessage] = None) -> dict:
        try:
            llm = ChatOpenAI(
                model_name=MODEL_NAME,
                openai_api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                temperature=0.1
            )

            clean_query = rewrite_query(llm, query, messages or [])
            logging.info("Clean query: %s", clean_query)

            vector_store = PineconeVectorStore(
                embedding=CustomEmbeddings(),
                index_name="gorgia",
                namespace="gorgia_products",
                pinecone_api_key=PINECONE_API_KEY
            )
            results = vector_store.similarity_search(clean_query, k=30)

            if not results:
                return {
                    "type": "product_list",
                    "message": "სამწუხაროდ აღნიშნული პროდუქტი ვერ მოიძებნა.",
                    "products": []
                }
            products = []
            for doc in results:
                product = {
                    "name": doc.metadata.get('name', 'N/A'),
                    "category": doc.metadata.get('category', 'N/A'),
                    "price": doc.metadata.get('price', 'N/A'),
                    "description": doc.metadata.get('description', '')
                }
                products.append(product)

            filter_prompt = f"""
                        დავალება: გაფილტრე პროდუქტები მომხმარებლის მოთხოვნის მიხედვით და დააბრუნე მხოლოდ ყველაზე შესაბამისი 10 პროდუქტის ინდექსი.

                        მომხმარებლის მოთხოვნა: {clean_query}
                        
                        ხელმისაწვდომი პროდუქტები: {str(products)}

                        მკაცრი წესები:
                        1. დააბრუნე მხოლოდ ის პროდუქტები, რომლებიც ზუსტად შეესაბამება მოთხოვნილ კატეგორიას
                          - მაგ: თუ მოთხოვნილია "ტელევიზორი", მხოლოდ ტელევიზორები უნდა დაბრუნდეს
                        2. თუ მოთხოვნილია კონკრეტული ბრენდი, პირველ რიგში დააბრუნე ამ ბრენდის პროდუქტები
                          - მაგ: "Samsung მაცივარი" - ჯერ Samsung-ის მაცივრები, შემდეგ სხვა მაცივრები
                        3. მაქსიმუმ 10 პროდუქტი უნდა დაბრუნდეს
                        4. დააბრუნე მხოლოდ ინდექსების სია JSON ფორმატში, მაგ: [0, 3, 5, 7]
                        5. თუ მოთხოვნილი კატეგორიის პროდუქტი ვერ მოიძებნა, დააბრუნე ცარიელი სია: []

                        დააბრუნე მხოლოდ ინდექსების სია, არანაირი დამატებითი ტექსტი.
                        """
            response = llm.invoke([SystemMessage(content=filter_prompt)])

            try:
                import json
                filtered_indices = json.loads(response.content)
                filtered_indices = filtered_indices[:10]
                filtered_products = []
                for idx in filtered_indices:
                    doc = results[idx]
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
                    filtered_products.append(product)

                return {
                    "type": "product_list",
                    "message": f"თქვენს მიერ მოთხოვნილი პროდუქტები:",
                    "products": filtered_products
                }

            except json.JSONDecodeError:
                logging.error(f"Error parsing LLM response: {response.content}")
                return {
                    "type": "error",
                    "message": "სამწუხაროდ, პროდუქტების ფილტრის დროს დაფიქსირდა შეცდომა.",
                    "products": []
                }
        except Exception as e:
            logging.error(f"Error in search_products: {str(e)}")
            return {
                "type": "error",
                "message": "სამწუხაროდ, პროდუქტების ძიებისას დაფიქსირდა შეცდომა.",
                "products": []
            }


class SearchDocsTool(BaseTool):
    name: str = "search_docs"
    description: str = "Search for documentation and general information"
    return_direct: bool = True

    def _run(self, query: str, messages: List[BaseMessage] = None) -> dict:
        try:
            llm = ChatOpenAI(
                model_name=MODEL_NAME,
                openai_api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                temperature=0
            )

            clean_query = rewrite_query(llm, query, messages or [])
            logging.info("Clean query:", clean_query)

            vector_store = PineconeVectorStore(
                embedding=CustomEmbeddings(),
                index_name="gorgia",
                namespace="gorgia_docs",
                pinecone_api_key = PINECONE_API_KEY

            )
            results = vector_store.similarity_search(clean_query, k=3)
            docs_content = "\n".join([f"{i}:\n{doc.page_content}" for i, doc in enumerate(results)])

            return format_docs_response(llm, query, docs_content)

        except Exception as e:
            return {
                "type": "error",
                "message": f"დოკუმენტის ძიების შეცდომა: {str(e)}"
            }


def rewrite_query(llm: ChatOpenAI, user_input: str, message_history: List[BaseMessage] = None) -> str:
    if not message_history:
        message_history = []

    history_text = ""
    if message_history:
        history_messages = []
        for msg in message_history[-6:]:
            if isinstance(msg, HumanMessage):
                history_messages.append(f"მომხმარებელი: {msg.content}")
            elif isinstance(msg, SystemMessage):
                history_messages.append(f"სისტემა: {msg.content}")
        history_text = "\n".join(history_messages)

    prompt = f"""
        მიზანი: მომხმარებლის შეკითხვის გარდაქმნა ოპტიმიზირებულ საძიებო ტექსტად.

        კონტექსტი:
        - წინა საუბრის ისტორია: {history_text}

        - მიმდინარე შეკითხვა: "{user_input}"

        ინსტრუქციები:
        1. გარდაქმენი შეკითხვა მოკლე, მიზნობრივ საძიებო ტერმინებად
        2. მოაშორე ზედმეტი სიტყვები და კონტექსტი
        3. აუცილებლად გაითვალისწინე წინა კონტექსტი თუ საჭიროა სპეციფიკური პროდუქტების შემთხვევაში(იხილე მაგალითი ქვემოთ)
        4. დატოვე მხოლოდ არსებითი საძიებო ტერმინები და გამოიყენე მხოლოდ არსებითი სახელები და მთავარი საკვანძო სიტყვები

        მაგალითები:

        User: "გაზქურა მინდა სამზარეულოში დასადგმელად"
        Clean Query: "გაზქურა"

        User: გაზქურა მინდა?
        System: [გაზქურები ცოდნის ბაზიდან]
        User: KUMTEL ის მინდა
        System: საძიებო ტექსტი - KUMTEL გაზქურა
        
        User: სკამი მინდა
        System: [სკამები ცოდნის ბაზიდან]
        User: ყვითელი
        System: საძიებო ტექსტი - ყვითელი სკამი

        წესები:
        1. არ ჩასვა ემოციური ან აღწერილობითი სიტყვები
        2. არ გამოიყენო ზმნები ან ზედსართავი სახელები
        3. შეინარჩუნე მხოლოდ მნიშვნელოვანი სპეციფიკური ტერმინები

        საძიებო ტექსტი: """

    response = llm.invoke([SystemMessage(content=prompt)])
    return response.content.strip()


def format_docs_response(llm: ChatOpenAI, query: str, docs_content: str) -> dict:
    format_prompt = f"""
    მოცემულია მომხმარებლის კითხვა და საინფორმაციო ბაზიდან მოძიებული ინფორმაცია.
    გთხოვთ, დააფორმატოთ პასუხი ისე, რომ:
    1. იყოს პირდაპირი და მარტივი პასუხი კითხვაზე
    2. შეინარჩუნოს ყველა მნიშვნელოვანი დეტალი
    3. კითხვაზე გაეცი ლოგიკური პასუხი

    კითხვა: {query}
    მოძიებული ინფორმაცია: {docs_content}
    """

    response = llm.invoke([SystemMessage(content=format_prompt)])
    return {
        "type": "text",
        "message": response.content
    }

def determine_next_step(state: GorgiaState) -> GorgiaState:
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.1
    )
    context = ""
    for msg in state['messages'][-3:]:
        if isinstance(msg, HumanMessage):
            context += f"მომხმარებელი: {msg.content}\n"
        elif isinstance(msg, SystemMessage):
            context += f"სისტემა: {msg.content}\n"

    decision_prompt = f"""
    გააანალიზე მომხმარებლის შეკითხვა და დააბრუნე შესაბამისი მოქმედება.
    
    წინა საუბრის კონტექსტი: "{context}"
    შეკითხვა: "{state['current_query']}"

    მკაცრი წესები:
    1. დააბრუნე "search_products" თუ:
       - შეკითხვა ეხება პროდუქტის ძიებას
       - შეკითხვა ეხება უკვე მოძებნილი პროდუქტების შესახებ ახალ საძიებო ფილტრს(მაგ. ნაჩვენებია მაცივრები და მომხმარებელი კითხულობს კონკრეტული ბრენდის მაცივრებზე)
       - ნახსენებია კონკრეტული პროდუქტი (მაგ: გაზქურა, მაცივარი)
       - ნახსენებია ბრენდი (მაგ: KUMTEL)
       - მომხმარებელს სურს პროდუქტის ნახვა ან ყიდვა

    2. დააბრუნე "search_docs" თუ:
       - შეკითხვა ეხება მიწოდების სერვისს
       - შეკითხვა ეხება გადახდის მეთოდებს
       - შეკითხვა ეხება გარანტიას
       - შეკითხვა ეხება კომპანიის ზოგად ინფორმაციას

    3. დააბრუნე "process_chat" თუ:
       - შეკითხვა არის მისალმება
       - შეკითხვა არის დამშვიდობება
       - მომხმარებელი მადლობას გიხდის
       - არის ზოგადი საუბარი

    დააბრუნე მხოლოდ ერთი შემდეგი მნიშვნელობებიდან: "search_products", "search_docs", "process_chat"
    დააბრუნე მხოლოდ მნიშვნელობა, არანაირი დამატებითი ტექსტი.
    პასუხი: """

    response = llm.invoke([SystemMessage(content=decision_prompt)])
    state['next_step'] = response.content.strip().lower()
    logging.info(f"Next step determined: {state['next_step']}")
    return state


def process_products(state: GorgiaState) -> GorgiaState:
    """Process product search requests."""
    logging.info("Processing product search...")
    tool = SearchProductsTool()
    try:
        result = tool.run(state['current_query'], state['messages'])
        state['response'] = result
        logging.info(f"Product search result: {result['type']}")
        return state
    except Exception as e:
        logging.error(f"Error in process_products: {str(e)}")
        state['response'] = {
            "type": "error",
            "message": f"პროდუქტების ძიებისას მოხდა შეცდომა: {str(e)}"
        }
        return state


def process_docs(state: GorgiaState) -> GorgiaState:
    tool = SearchDocsTool()
    response = tool.run(state['current_query'], state['messages'])
    state['response'] = response
    return state


def process_chat(state: GorgiaState) -> GorgiaState:
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.1
    )

    system_prompt = """
    GORGIA არის უმსხვილესი კორპორაცია სამშენებლო და სარემონტო სფეროში კავკასიაში.    
    თქვენ ხართ Gorgia-ს სპეციალიზირებული ასისტენტი.
    """

    messages = [SystemMessage(content=system_prompt)] + state['messages']
    response = llm.invoke(messages)

    state['response'] = {
        "type": "text",
        "message": response.content
    }
    return state


def create_gorgia_graph() -> StateGraph:
    workflow = StateGraph(GorgiaState)

    workflow.add_node("determine_next_step", determine_next_step)
    workflow.add_node("process_products", process_products)
    workflow.add_node("process_docs", process_docs)
    workflow.add_node("process_chat", process_chat)

    workflow.set_entry_point("determine_next_step")

    workflow.add_conditional_edges(
        "determine_next_step",
        lambda x: x["next_step"],
        {
            "search_products": "process_products",
            "search_docs": "process_docs",
            "process_chat": "process_chat"
        }
    )

    for node in ["process_products", "process_docs", "process_chat"]:
        workflow.add_edge(node, END)

    return workflow.compile()


class GorgiaAgent:
    def __init__(self):
        self.graph = create_gorgia_graph()
        self.state: GorgiaState = {
            "messages": [],
            "next_step": "",
            "current_query": "",
            "products": [],
            "docs": "",
            "response": {}
        }

    def run(self, user_input: str) -> dict:
        self.state["messages"].append(HumanMessage(content=user_input))
        self.state["current_query"] = user_input

        try:
            result = self.graph.invoke(self.state)
            self.state = result
            if result["response"]:
                if isinstance(result["response"], dict):
                    self.state["messages"].append(
                        SystemMessage(content=result["response"].get("message", ""))
                    )
                else:
                    self.state["messages"].append(
                        SystemMessage(content=str(result["response"]))
                    )
            if len(self.state["messages"]) > 6:
                self.state["messages"] = []

            return result["response"] or {
                "type": "error",
                "message": "დაფიქსირდა შეცდომა დამუშავებისას."
            }

        except Exception as e:
            print(f"Error in run method: {str(e)}")
            return {
                "type": "error",
                "message": f"დაფიქსირდა შეცდომა: {str(e)}"
            }

    def clear_history(self):
        self.state["messages"] = []


def create_gorgia_agent():
    return GorgiaAgent()
