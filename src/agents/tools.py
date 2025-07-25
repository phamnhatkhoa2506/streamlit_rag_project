from typing import Optional, List
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from enums.memory_type import MemoryType
from memory.store import store_memory, SYSTEM_USER_ID, retrieve_memories
from memory.search_index import long_term_memory_index


@tool
def store_memory_tool(
    content: str,
    memory_type: MemoryType,
    metadata: Optional[str] = None,
    config: Optional[RunnableConfig] = None
) -> str:
    """
        Store a long-term memory in the system.

        Use this tool to save important information about user preferences,
        experiences, or general knowledge that might be useful in future
        interactions.
    """

    config = config or RunnableConfig()
    user_id = config.get("user_id", SYSTEM_USER_ID)
    thread_id = config.get("thread_id")

    try:
        store_memory(
            long_term_memory_index,
            content,
            memory_type,
            user_id,
            thread_id,
            metadata = str(metadata) if metadata else None
        )
        return f"Successfully stored {memory_type} memory: {content}"
    except Exception as e:
        return f"Error storing memory: {str(e)}"

    
@tool
def retrieve_memories_tool(
    query: str,
    memory_types: List[MemoryType],
    limit: int = 5,
    config: Optional[RunnableConfig] = None
) -> str:
    """
        Retrieve long-term memories relevant to the query.

        Use this tool to access previously stored information about user
        preferences, experiences, or general knowledge.
    """
    config = config or RunnableConfig()
    user_id = config.get("user_id", SYSTEM_USER_ID)

    try:
        stored_memories = retrieve_memories(
            long_term_memory_index,
            query,
            memory_types,
            user_id,
            limit=limit,
            distance_threshold=0.3
        )

        responses = []

        if stored_memories:
            responses.append("Long-term memories:")
            for memory in stored_memories:
                responses.append(f"- [{memory.memory_type}] {memory.content}")

        return " ".join(responses) if responses else "No relevant memories found."
    
    except Exception as e:
        return f"Error retrieving memories: {str(e)}"


tools = [store_memory_tool, retrieve_memories_tool]
