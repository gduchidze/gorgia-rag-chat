from typing import  List
from langchain_core.messages import  BaseMessage
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.tools import BaseTool
from helpers import rewrite_query, format_docs_response
from config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL, PINECONE_API_KEY, CustomEmbeddings, logger


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
            logger.info(f"Clean query: {clean_query}")

            vector_store = PineconeVectorStore(
                embedding=CustomEmbeddings(),
                index_name="gorgia",
                namespace="gorgia_products",
                pinecone_api_key=PINECONE_API_KEY

            )
            results = vector_store.similarity_search(clean_query, k=10)

            if not results:
                return {
                    "type": "product_list",
                    "message": "სამწუხაროდ აღნიშნული პროდუქტი ვერ მოიძებნა.",
                    "products": []
                }

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

            return {
                "type": "product_list",
                "message": f"თქვენს მიერ მოთხოვნილი პროდუქტები:",
                "products": products
            }

        except Exception as e:
            logger.info(f"Error in search_products: {str(e)}")
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
            logger.info("Clean query:", clean_query)

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