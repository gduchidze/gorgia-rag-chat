from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from chat.main import SearchProductsTool,process_docs
from config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL, GorgiaState, logger


def determine_next_step(state: GorgiaState) -> GorgiaState:
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.1
    )

    decision_prompt = f"""
    გააანალიზე მომხმარებლის შეკითხვა და დააბრუნე შესაბამისი მოქმედება.

    შეკითხვა: "{state['current_query']}"

    მკაცრი წესები:
    1. დააბრუნე "search_products" თუ:
       - შეკითხვა ეხება პროდუქტის ძიებას
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
    logger.info(f"Next step determined: {state['next_step']}")
    return state


def process_products(state: GorgiaState) -> GorgiaState:
    """Process product search requests."""
    logger.info("Processing product search...")
    tool = SearchProductsTool()
    try:
        result = tool.run(state['current_query'], state['messages'])
        state['response'] = result
        logger.info(f"Product search result: {result['type']}")
        return state
    except Exception as e:
        logger.info(f"Error in process_products: {str(e)}")
        state['response'] = {
            "type": "error",
            "message": f"პროდუქტების ძიებისას მოხდა შეცდომა: {str(e)}"
        }
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
            logger.error(f"Error in run method: {str(e)}")
            return {
                "type": "error",
                "message": f"დაფიქსირდა შეცდომა: {str(e)}"
            }

    def clear_history(self):
        self.state["messages"] = []


def create_gorgia_agent():
    return GorgiaAgent()
