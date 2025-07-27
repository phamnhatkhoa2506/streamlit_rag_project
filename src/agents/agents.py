from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.redis import RedisSaver

from agents.tools import tools
from llms.chat_models import main_chat_model
from rag.chain import chain
from utils.redis_connection import redis_client


redis_saver = RedisSaver(redis_client=redis_client)
redis_saver.setup()


travel_agent = create_react_agent(
    model=main_chat_model,
    tools=tools,
    checkpointer=redis_saver,
    prompt=SystemMessage(
        content="""
        You are a travel assistant helping users plan their trips. You remember user preferences
        and provide personalized recommendations based on past interactions.

        You have access to the following types of memory:
        1. Short-term memory: The current conversation thread
        2. Long-term memory:
           - Episodic: User preferences and past trip experiences (e.g., "User prefers window seats")
           - Semantic: General knowledge about travel destinations and requirements

        Your procedural knowledge (how to search, book flights, etc.) is built into your tools and prompts.

        Always be helpful, personal, and context-aware in your responses.
        """
    ),
)
