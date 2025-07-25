from ulid import ULID
from datetime import datetime
from typing import Optional, List
from pydantic import Field, BaseModel
from enums.memory_type import MemoryType


class Memory(BaseModel):
    content: str
    metadata: str
    memory_type: MemoryType


class StoredMemory(Memory):
    id: str
    memory_id: ULID = Field(default_factory=lambda: ULID())
    created_at: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    memory_type: Optional[MemoryType] = None


class Memories:
    memories: List[Memory]