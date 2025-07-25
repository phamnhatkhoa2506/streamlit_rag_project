import torch
from typing import Literal
from uuid import uuid4
from pinecone import Pinecone
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_core.stores import InMemoryStore
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import ServerlessSpec


class VectorStore:
    def __init__(
        self,
        name: str,
        documents: list[Document],
        storedb: Literal['chroma', 'faiss', 'pinecone'] = 'chroma',
        **kwargs
    ) -> None:
        '''
            Parameters:
                name: str - The name of the vector store
                documents: list[Document] - The documents to add to the vector store
                storedb: Literal['chroma', 'faiss', 'pinecone'] - The type of vector store to use
                **kwargs - Additional keyword arguments to pass to the vector store
        '''

        self.name = name
        self.documents = documents

        # Embedding model
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Indices for documents
        self.ids = [str(uuid4()) for _ in range(len(documents))]

        if storedb == 'chroma':
            # Chroma vector store
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                ids=self.ids,
                collection_name=name,
                embedding=self.embedding_model,
                collection_metadata={
                    key: value for key, value in kwargs.items()
                },
                persist_directory=f"./chroma_db/{name}"
            )
        elif storedb == 'faiss':
            # FAISS vector store
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                index=self.name,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={},
            )
        elif storedb == 'pinecone':
            # Pinecone vector store
            pc = Pinecone()
            if not pc.has_index(self.name):
                pc.create_index(
                    name=self.name,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            index = pc.Index(self.name)

            self.vectorstore = PineconeVectorStore(
                index=index,
                embedding=self.embedding_model,
            )
            self.vectorstore.add_documents(documents, ids=self.ids)
        else:       
            raise ValueError(f"Invalid vector store: {storedb}")

    def add_documents(
        self,
        documents: list[Document],
    ):
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vectorstore.add_documents(documents, ids=uuids)

    def search(
        self,
        query: str,
        search_kwargs: dict = {
            'k': 10,
        }
    ):
        return self.vectorstore.similarity_search(
            query=query,
            **search_kwargs
        )

    def similarity_search_with_score(
        self,
        query: str,
        search_kwargs: dict = {
            'k': 10,
        }
    ):
        return self.vectorstore.similarity_search_with_score(
            query=query,
            **search_kwargs
        )

    def get_compression_retriever(
        self,
        search_type: Literal['similarity', 'similarity_score_threshold', 'mmr'] = 'similarity',
        search_kwargs: dict = {
            'k': 10,
            'score_threshold': 0.5,
        }
    ) -> ContextualCompressionRetriever:
        torch.cuda.empty_cache()
        retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

        # Compressor
        compressor = RankLLMRerank(top_n=3, model="rank_zephyr")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        del compressor

        return compression_retriever

    def get_compression_multivector_retriever(self):
        retriever = MultiVectorRetriever(
            docstore=InMemoryStore(),
            vectorstore=self.vectorstore,
            byte_store=InMemoryByteStore(),
            id_key='doc_id',
        )

        retriever.docstore.mset(list(zip(self.ids, self.documents)))

        compressor = RankLLMRerank(top_n=3, model="rank_zephyr")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        del compressor

        return compression_retriever


        
