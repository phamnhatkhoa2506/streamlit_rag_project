from ulid import ULID
from typing import Optional, Union, List
from datetime import datetime
from redisvl.query import VectorRangeQuery
from redisvl.query.filter import Tag
from redisvl.index import SearchIndex
from enums.memory_type import MemoryType
from models.memory import StoredMemory
from llms.embedding_models import redis_embedding_model


SYSTEM_USER_ID = "system"


def similar_memory_exists(
    long_term_memory_index: SearchIndex,
    content: str,
    memory_type: MemoryType,
    user_id: str = SYSTEM_USER_ID,
    thread_id: Optional[str] = None,
    distance_threshold: float = 0.1
) -> bool:
    content_embedding = redis_embedding_model.embed_query(content)

    filters = (Tag("user_id") == user_id) & (Tag("memory_type") == memory_type)

    if thread_id:
        filters = filters & (Tag("thread_id") == thread_id)

    vector_query = VectorRangeQuery(
        vector=content_embedding,
        num_results=1,
        vector_field_name="embedding",
        filter_expression=filters,
        distance_threshold=distance_threshold,
        return_fields=["id"]
    )

    results = long_term_memory_index.query(vector_query)
   
    if results:
        return True

    return False


def store_memory(
    long_term_memory_index: SearchIndex,
    content: str,
    memory_type: MemoryType,
    user_id: str = SYSTEM_USER_ID,
    thread_id: Optional[str] = None,
    metadata: Optional[str] = None
) -> None:
    if metadata is None:
        metadata = "{}"

    if similar_memory_exists(
        long_term_memory_index,
        content,
        memory_type,
        user_id,
        thread_id
    ):
        return
    
    embedding = redis_embedding_model.embed_query(content)
    
    memory_data = {
        "user_id": user_id or SYSTEM_USER_ID,
        "content": content,
        "memory_type": memory_type.value,
        "metadata": metadata,
        "created_at": datetime.now().isoformat(),
        "memory_id": str(ULID()),
        "embedding": embedding,
        "thread_id": thread_id
    }

    try:
        long_term_memory_index.load([memory_data])
    except Exception as e:
        return


def retrieve_memories(
    long_term_memory_index: SearchIndex,
    query: str,
    memory_type: Union[Optional[MemoryType], List[MemoryType]] = None,
    user_id: str = SYSTEM_USER_ID,
    thread_id: Optional[str] = None,
    distance_threshold: float = 0.1,
    limit: int = 5
) -> List[StoredMemory]:

    vector_query = VectorRangeQuery(
        vector=redis_embedding_model.embed_query(query),
        return_fields=[
            "content",
            "metadata",
            "memory_type",
            "created_at",
            "memory_id",
            "user_id",
            "thread_id"
        ],
        num_results=limit,
        vector_field_name="embedding",
        dialect=2,
        distance_threshold=distance_threshold
    )

    base_filters = [f"@user_id:{{{user_id or SYSTEM_USER_ID}}}"]

    if memory_type:
        if isinstance(memory_type, list):
            base_filters.append(f"@memory_type: {{{'|'.join([mt.value for mt in memory_type])}}}")
        else:
            base_filters.append(f"@memory_type: {{{memory_type.value}}}")

    if thread_id:
        base_filters.append(f"@thread_id: {{{thread_id}}}")

    vector_query.set_filter(" ".join(base_filters))

    results = long_term_memory_index.query(vector_query)

    memories = []
    for doc in results:
        try:
            memory = StoredMemory(
                id=doc["id"],
                memory_id=doc["memory_id"],
                created_at=doc["created_at"],
                user_id=doc["user_id"],
                thread_id=doc["thread_id"],
                memory_type=doc["memory_type"],
                content=doc["content"],
                metadata=doc["metadata"]
            )
            memories.append(memory)
        except Exception as e:
            continue

    return memories