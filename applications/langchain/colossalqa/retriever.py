# from langchain.schema.retriever import BaseRetriever, Document
# from langchain.callbacks.manager import CallbackManagerForRetrieverRun
# from typing import List, Callable, Dict, Any, Union

# class CustomRetriever(BaseRetriever):
#     retriever: List[BaseRetriever] = None
#     sql_db_chains = []
#     k = 3
#     rephrase_handler:Callable = None
#     buffer: Dict = []
#     buffer_size: int = 5

#     def set_retriever(self, retriever:Union[BaseRetriever, List[BaseRetriever]]):
#         '''update retriever. Useful when you want to change the supporting documents'''
#         self.retriever = [retriever] if isinstance(retriever, BaseRetriever) else retriever

#     def set_sql_database_chain(self, db_chains):
#         '''
#         set sql agent chain to retrieve information from sql database
#         Not used in this version
#         '''
#         self.sql_db_chains = db_chains

#     def set_rephrase_handler(self, handler:Callable=None):
#         '''
#         Set a handler to preprocess the input str before feed into the retriever
#         '''
#         self.rephrase_handler = handler

#     def _get_relevant_documents(
#         self, query: str, *, run_manager: CallbackManagerForRetrieverRun=None, 
#         score_threshold: float = None
#     ) -> List[Document]:
#         '''
#         This function is called by the retriever to get the relevant documents.
#         recent vistied queries are stored in buffer, if the query is in buffer, return the documents directly
        
#         Args:
#             query: the query to be searched
#             run_manager: the callback manager for retriever run
#         Returns:
#             documents: the relevant documents
#         '''
#         for buffered_doc in self.buffer:
#             if buffered_doc[0] == query:
#                 return buffered_doc[1]
#         query_ = str(query)
#         # Use your existing retriever to get the documents
#         if self.rephrase_handler:
#             query = self.rephrase_handler(query)
#         documents = []
#         for retriever in self.retriever:
#             # retrieve documents from each retriever
#             k = retriever.search_kwargs['k'] if 'k' in retriever.search_kwargs else self.k
#             documents.extend(retriever.vectorstore.similarity_search_with_relevance_scores(query, k, score_threshold=score_threshold))
#         # return the top k documents among all retrievers
#         documents = sorted(documents, key=lambda x: x[1], reverse=True)[:self.k]
#         documents = [doc[0] for doc in documents]
#         # retrieve documents from sql database (not applicable for the local chains)
#         for sql_chain in self.sql_db_chains:
#             documents.append(Document(page_content = f"Query: {query}  Answer: {sql_chain.run(query)}", metadata={"source": "sql_query"}))
#         if len(self.buffer)<self.buffer_size:
#             self.buffer.append([query_, documents])
#         else:
#             self.buffer.pop(0)
#             self.buffer.append([query_, documents])
#         print("retrieved documents:")
#         print(documents)  
#         return documents

from langchain.schema.retriever import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List, Callable, Dict, Any, Union
from colossalqa.utils import create_empty_sql_database, destroy_sql_database
from langchain.indexes import SQLRecordManager 
from langchain.embeddings.base import Embeddings
from langchain.indexes import index
from collections import defaultdict
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.chroma import Chroma
import hashlib

class CustomRetriever(BaseRetriever):
    vector_stores: Dict[str, VectorStore] = {}
    sql_index_database: Dict[str, str] = {}
    record_managers: Dict[str, SQLRecordManager]={}
    sql_db_chains = []
    k = 3
    rephrase_handler:Callable = None
    buffer: Dict = []
    buffer_size: int = 5

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embeddings: Embeddings,
        **kwargs: Any,
    ) -> BaseRetriever:
        k = kwargs.pop('k', 3)
        cleanup = kwargs.pop('cleanup', 'incremental')
        mode = kwargs.pop('mode', 'by_source')
        ret = cls(k=k)
        ret.add_documents(documents, embedding=embeddings, cleanup=cleanup, mode=mode)
        return ret

    def add_documents(self, docs:Dict[str, Document]=[], cleanup:str='incremental', mode:str='by_source', embedding:Embeddings=None) -> None:
        '''
        add documents to retriever
        Args:
            docs: the documents to add
            cleanup: choose from "incremental" (update embeddings, skip existing embeddings) and "full" (destory and rebuild retriever)
            mode: choose from "by source" (documents are grouped by source) and "merge" (documents are merged into one vector store)
        '''
        if cleanup == "full":
            # cleanup
            for k in self.vector_stores:
                if self.sql_index_database[k]:
                    destroy_sql_database(self.sql_index_database[k])
            self.vector_stores = {}
            self.sql_index_database = {}
            self.record_managers = {}
        # add documents
        data_by_source = defaultdict(list)
        if mode == "by_source":
            for doc in docs:
                data_by_source[doc.metadata['source']].append(doc)
        elif mode == "merge":
            data_by_source['merged'] = docs
        for source in data_by_source:
            if source not in self.vector_stores:
                _, sql_path = create_empty_sql_database(f"sqlite:///{source.replace('.','_')}.db")
                self.vector_stores[source] = Chroma(embedding_function=embedding, 
                        collection_name=hashlib.sha1(source.encode()).hexdigest())
                self.sql_index_database[source] = sql_path
                self.record_managers[source] = SQLRecordManager(source, db_url=sql_path)
                self.record_managers[source].create_schema()
            index(
                data_by_source[source],
                self.record_managers[source],
                self.vector_stores[source],
                cleanup=cleanup,
                source_id_key="source"
            )

    def __del__(self) -> None:
        self.add_documents([], cleanup='full')

    def set_sql_database_chain(self, db_chains) -> None:
        '''
        set sql agent chain to retrieve information from sql database
        Not used in this version
        '''
        self.sql_db_chains = db_chains

    def set_rephrase_handler(self, handler:Callable=None) -> None:
        '''
        Set a handler to preprocess the input str before feed into the retriever
        '''
        self.rephrase_handler = handler

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun=None, 
        score_threshold: float = None, return_scores: bool=False
    ) -> List[Document]:
        '''
        This function is called by the retriever to get the relevant documents.
        recent vistied queries are stored in buffer, if the query is in buffer, return the documents directly
        
        Args:
            query: the query to be searched
            run_manager: the callback manager for retriever run
        Returns:
            documents: the relevant documents
        '''
        for buffered_doc in self.buffer:
            if buffered_doc[0] == query:
                return buffered_doc[1]
        query_ = str(query)
        # Use your existing retriever to get the documents
        if self.rephrase_handler:
            query = self.rephrase_handler(query)
        documents = []
        for k in self.vector_stores:
            # retrieve documents from each retriever
            vectorstore = self.vector_stores[k]
            documents.extend(vectorstore.similarity_search_with_relevance_scores(query, self.k, score_threshold=score_threshold))
        # return the top k documents among all retrievers
        documents = sorted(documents, key=lambda x: x[1], reverse=True)[:self.k]
        if return_scores:
            # return score
            for doc in documents:
                doc[0].metadata['score'] = doc[1]
        documents = [doc[0] for doc in documents]
        # retrieve documents from sql database (not applicable for the local chains)
        for sql_chain in self.sql_db_chains:
            documents.append(Document(page_content = f"Query: {query}  Answer: {sql_chain.run(query)}", metadata={"source": "sql_query"}))
        if len(self.buffer)<self.buffer_size:
            self.buffer.append([query_, documents])
        else:
            self.buffer.pop(0)
            self.buffer.append([query_, documents])
        print("retrieved documents:")
        print(documents)  
        return documents