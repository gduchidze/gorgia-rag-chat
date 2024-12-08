import logging
import os
from typing import TypedDict, Dict, List

from dotenv import load_dotenv
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain_core.messages import BaseMessage
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


OPENAI_API_KEY = "sk-MckIJPkrp42Ev4_EBkj6aQ"
OPENAI_BASE_URL = "https://api.ailab.ge"
MODEL_NAME = "tbilisi-ai-lab-2.0"
DEEPINFRA_API_TOKEN = "hHnQ6vbPMRKj7eCs5IG6QmjFTyYjVccW"
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


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

# class CustomEmbeddings(OpenAIEmbeddings):
#     def __init__(self):
#         super().__init__(openai_api_key=EMBEDDINGS_API_KEY)
#         self.client = OpenAI(api_key=EMBEDDINGS_API_KEY, base_url=EMBEDDINGS_API_BASE)
#         self.model = EMBEDDINGS_MODEL