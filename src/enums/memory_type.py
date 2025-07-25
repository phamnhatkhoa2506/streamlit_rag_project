from enum import Enum


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"