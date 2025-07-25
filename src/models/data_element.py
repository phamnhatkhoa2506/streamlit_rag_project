from pydantic import BaseModel
from src.enums.data_element_type import DataElementType


class DataElement(BaseModel):
    type: DataElementType
    text: str