import os
import re
from typing import Dict, Any, List
# from itertools import chain
# from tqdm import tqdm
from langchain.schema import Document
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from unstructured.partition.pdf import partition_pdf
# from unstructured.partition.docx import partition_docx
# from unstructured.partition.text import partition_text
# from unstructured.partition.html import partition_html
# from unstructured.partition.ppt import partition_ppt

# from enums.data_element_type import DataElementType
# from models.data_element import DataElement
from loggers.logger import logger


# class DocumentLoader(object):
#     def __init__(
#         self,
#         seperators: List[str] = ['\n\n', '\n', ' ', ''],
#         split_kwargs: Dict[str, Any] = {
#             "chunk_size": 300,
#             "chunk_overlap": 0
#         }
#     ) -> None:
#         '''
#             Parameters:
#                 seperators: List[str] - The separators to use for splitting the text
#                 split_kwargs: Dict[str, Any] - The keyword arguments to pass to the splitter
#         '''

#         self.partition_functions = {
#             "pdf": self.extract_pdf,
#             "docx": partition_docx,
#             "text": partition_text,
#             "html": partition_html,
#             "ppt": partition_ppt,
#         }

#         self.splitter = RecursiveCharacterTextSplitter(
#             separators=seperators,
#             **split_kwargs
#         )

#     def remove_non_utf8_characters(self, text: str) -> str:
#         '''
#             Remove non-UTF8 characters from text
#         '''

#         return re.sub(r'[^\x00-\x7F]+', '', text)

#     def get_documents(self, elements: List) -> List[Document]:
#         '''
#             Get documents from raw elements
#         '''
#         logger.info(f"Starting to get documents")

#         docs = [Document(page_content=element.text) for element in elements]

#         return docs

#     def get_data_elements(self, raw_elements: List[Any]) -> List[DataElement]:
#         '''
#             Get data elements from raw elements

#             Parameters:
#                 raw_elements: List[Any] - The raw elements to get data elements from
#         '''
        
#         logger.info(f"Starting to get data elements")
        
#         elements = []
#         for element in raw_elements:
#             text = self.remove_non_utf8_characters(element.text)
#             if "unstructured.documents.elements.CompositeElement" in str(type(element)):
#                 data_element = DataElement(
#                     type=DataElementType.TEXT,
#                     text=text
#                 )
#             elif "unstructured.documents.elements.Table" in str(type(element)):
#                 data_element = DataElement(
#                     type=DataElementType.TABLE,
#                     text=text
#                 )
#             elif "unstructured.documents.elements.Image" in str(type(element)):
#                 data_element = DataElement(
#                     type=DataElementType.IMAGE,
#                     text=text
#                 )
#             else:
#                 # Default to text type for unknown elements
#                 data_element = DataElement(
#                     type=DataElementType.TEXT,
#                     text=text
#                 )

#             elements.append(data_element)

#         return elements

#     def extract_pdf(self, file_path: str) -> List[Any]:
#         '''
#             Extracts pdf data from a file

#             Parameters:
#                 file_path: str - The path to the file to extract pdf data from
#         '''

#         return partition_pdf(
#             filename=file_path,
#             chunking_strategy="by_title",
#             max_characters=1800,
#             new_after_n_chars=1500,
#             combine_text_under_n_chars=1000,
#         )

#     def load_documents(self, dir_path: str):
#         '''
#             Loads documents from a directory

#             Parameters:
#                 dir_path: str - The path to the directory containing the files
#         '''
        
#         if not os.path.exists(dir_path):
#             raise FileNotFoundError(f"Directory not found: {dir_path}")

#         list_of_files = os.listdir(dir_path)
#         logger.info(f"List of files: {list_of_files}")

#         file_paths = [
#             os.path.join(dir_path, file)
#                 for file in list_of_files
#         ]
#         file_names = [
#             os.path.basename(file)
#                 for file in file_paths
#         ]

#         documents = []
    
#         for i, file_path in tqdm(enumerate(file_paths)):
#             ext_part = file_names[i].split(".")[-1]
#             logger.info(f"Document: {file_names[i]}")
#             logger.info(f"Ext part: {ext_part}")
            
#             try:
#                 if ext_part not in self.partition_functions:
#                     logger.info(f"Unsupported file extension: {ext_part}")
#                     continue
                    
#                 raw_elements = self.partition_functions[ext_part](file_path)

#                 # logger.info(f"Raw elements: {len(raw_elements)}")

#                 elements = self.get_data_elements(raw_elements)
#                 docs = self.get_documents(elements)

#                 documents.extend(docs)

#                 logger.info(f"Docs: {len(docs)}")
#                 # doc_split = self.splitter.split_documents(doc_loaded)
#             except Exception as e:
#                 logger.error(f"Error processing file {file_path}: {e}")
#                 continue

#         logger.info(f"Documents: {len(documents)}")

#         return documents


class DocumentLoader(object):
    def __init__(
        self,
        seperators: List[str] = ['\n\n', '\n', ' ', ''],
        split_kwargs: Dict[str, Any] = {
            "chunk_size": 300,
            "chunk_overlap": 0
        }
    ) -> None:
        '''
            Parameters:
                seperators: List[str] - The separators to use for splitting the text
                split_kwargs: Dict[str, Any] - The keyword arguments to pass to the splitter
        '''

        self.splitter = RecursiveCharacterTextSplitter(
            separators=seperators,
            **split_kwargs
        )

    def remove_non_utf8_characters(self, text: str) -> str:
        '''
            Remove non-UTF8 characters from text
        '''

        return re.sub(r'[^\x00-\x7F]+', '', text)

    def load_documents(self, dir_path: str):
        '''
            Loads documents from a directory

            Parameters:
                dir_path: str - The path to the directory containing the files
        '''
        
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        list_of_files = os.listdir(dir_path)
        logger.info(f"List of files: {list_of_files}")

        # file_paths = [
        #     os.path.join(dir_path, file)
        #         for file in list_of_files
        # ]
    
        documents = PyPDFDirectoryLoader(dir_path).load()
        documents = [Document(page_content=self.remove_non_utf8_characters(document.page_content)) for document in documents]
        sentences = self.splitter.split_documents(documents)

        logger.info(f"Documents: {len(sentences)}")

        return sentences