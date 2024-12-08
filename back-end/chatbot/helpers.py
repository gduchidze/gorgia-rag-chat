from typing import List
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from chat.main import GorgiaState, SearchDocsTool


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
    3. გაითვალისწინე წინა კონტექსტი

    საძიებო ტექსტი:"""

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

def process_docs(state: GorgiaState) -> GorgiaState:
    tool = SearchDocsTool()
    response = tool.run(state['current_query'], state['messages'])
    state['response'] = response
    return state