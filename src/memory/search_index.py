import os
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema.schema import IndexSchema
from src.utils.redis_connection import redis_client


schema = {
    "index": {
        "name": "agent_memories",
        "prefix": "memory",
        "key_seperator": ":",
        "storage_type": "json"
    },
    "fields": [
        {"name": "content", "type": "text"},
        {"name": "memory_type", "type": "tag"},
        {"name": "metadata", "type": "text"},
        {"name": "created_at", "type": "text"},
        {"name": "user_id", "type": "tag"},
        {"name": "memory_id", "type": "tag"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "algorithm": "flat",
                "dims": 384,
                "distance_metric": "cosine",
                "datatype": "float32"
            }
        }
    ]
}


def create_search_index(redis_client: Redis) -> SearchIndex | None:
    try:
        memory_schema = IndexSchema.from_dict(schema)
        long_term_memory_index = SearchIndex(
            schema=memory_schema,
            redis_client=redis_client,
            validate_on_load=True
        )

        long_term_memory_index.create(overwrite=True)

        return long_term_memory_index
    except Exception as e:
        return None
    

long_term_memory_index = create_search_index(redis_client)