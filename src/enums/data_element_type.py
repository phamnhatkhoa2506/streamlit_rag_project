from enum import Enum


class DataElementType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"