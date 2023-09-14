"""Chain for question-answering against a vector database."""
from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import Any, Dict, List, Optional
import copy


from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)

from colossalqa.chain.retrieval_qa.stuff import CustomStuffDocumentsChain
from langchain.chains.llm import LLMChain
from colossalqa.chain.retrieval_qa.load_chain import load_qa_chain
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import Field
from langchain.schema import BaseRetriever, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains.retrieval_qa.base import BaseRetrievalQA


class CustomBaseRetrievalQA(BaseRetrievalQA):
    """Base class for question-answering chains."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseRetrievalQA:
        """Initialize from LLM."""
        if 'llm_kwargs' in kwargs:
            llm_kwargs = copy.deepcopy(kwargs['llm_kwargs'])
            del kwargs['llm_kwargs']
        else:
            llm_kwargs = {}
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        llm_chain = LLMChain(llm=llm, prompt=_prompt, callbacks=callbacks, llm_kwargs=llm_kwargs)
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="Context:\n{page_content}"
        )
        combine_documents_chain = CustomStuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
            callbacks=callbacks,
        )

        return cls(
            combine_documents_chain=combine_documents_chain,
            callbacks=callbacks,
            **kwargs,
        )

    @classmethod
    def from_chain_type(
        cls,
        llm: BaseLanguageModel,
        chain_type: str = "stuff",
        chain_type_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> BaseRetrievalQA:
        """Load chain from chain type."""
        if 'llm_kwargs' in kwargs:
            llm_kwargs = copy.deepcopy(kwargs['llm_kwargs'])
            del kwargs['llm_kwargs']
        else:
            llm_kwargs = {}

        _chain_type_kwargs = chain_type_kwargs or {}
        combine_documents_chain = load_qa_chain(
            llm, chain_type=chain_type, **_chain_type_kwargs, llm_kwargs=llm_kwargs
        )
        return cls(combine_documents_chain=combine_documents_chain, **kwargs)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(question, run_manager=_run_manager)
        else:
            docs = self._get_docs(question)  # type: ignore[call-arg]
        kwargs = {k:v for k,v in inputs.items() if k in ['stop', 'temperature', 'top_k', 'top_p', 'max_new_tokens']}
        answer = self.combine_documents_chain.run(
            input_documents=docs, question=question, callbacks=_run_manager.get_child(), **kwargs
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._aget_docs).parameters
        )
        if accepts_run_manager:
            docs = await self._aget_docs(question, run_manager=_run_manager)
        else:
            docs = await self._aget_docs(question)  # type: ignore[call-arg]
        answer = await self.combine_documents_chain.arun(
            input_documents=docs, question=question, callbacks=_run_manager.get_child()
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}


class RetrievalQA(CustomBaseRetrievalQA):
    """Chain for question-answering against an index.

    Example:
        .. code-block:: python

            from langchain.llms import OpenAI
            from langchain.chains import RetrievalQA
            from langchain.faiss import FAISS
            from langchain.vectorstores.base import VectorStoreRetriever
            retriever = VectorStoreRetriever(vectorstore=FAISS(...))
            retrievalQA = RetrievalQA.from_llm(llm=OpenAI(), retriever=retriever)

    """

    retriever: BaseRetriever = Field(exclude=True)

    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        return self.retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )

    async def _aget_docs(
        self,
        question: str,
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        return await self.retriever.aget_relevant_documents(
            question, callbacks=run_manager.get_child()
        )

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "retrieval_qa"