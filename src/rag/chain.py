from typing import Any, List
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from rag.parser import OutputParser

class Chain():
    def __init__(
        self,
    ) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.5,
        )
        self.prompt = hub.pull("rlm/rag-prompt")
        self.parser = OutputParser()

    def get_chain(self, com_retriever: Any) -> Any:
        input_data = {
            "context": com_retriever | self.format_docs,
            "question": RunnablePassthrough()
        }

        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | self.parser
        )

        return rag_chain

    def get_qa_chain(self, com_retriever: Any) -> Any:
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=com_retriever,
            return_source_documents=True,
            verbose=True
        )

    def format_docs(self, docs: List[Any]) -> str:
        """
        Format documents into a single string
        """
        if not docs:
            return "No relevant documents found."
        
        formatted_docs = []
        for doc in docs:
            if hasattr(doc, 'page_content'):
                formatted_docs.append(doc.page_content)
            elif isinstance(doc, str):
                formatted_docs.append(doc)
            elif isinstance(doc, dict) and 'page_content' in doc:
                formatted_docs.append(doc['page_content'])
        
        return "\n\n".join(formatted_docs) 

def build_rag_chain():
    """
    Build and return a RAG chain. This is a stub; you should implement the actual retriever logic as needed.
    """
   
    com_retriever = None  
    chain = Chain()
    return chain.get_chain(com_retriever) 