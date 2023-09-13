'''
Script for English retrieval based conversation system backed by LLaMa2
'''
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import LLMChain
from colossalqa.chain.retrieval_qa.base import RetrievalQA
from langchain.vectorstores import Chroma
from colossalqa.local.llm import VllmLLM, CoatiAPI, CoatiLLM
from colossalqa.data_loader.document_loader import DocumentLoader
from colossalqa.retriever import CustomRetriever
from colossalqa.text_splitter import NeuralTextSplitter
from colossalqa.memory import ConversationBufferWithSummary
from colossalqa.prompt.prompt import PROMPT_RETRIEVAL_QA_EN, PROMPT_DISAMBIGUATE_EN

# vllm
# start the vllm server with
# python -m vllm.entrypoints.api_server --model "/path to model/Llama-2-7b-hf" --swap-space 16 --disable-log-requests --host localhost --port 8077 --max-num-seqs 256 --gpu-memory-utilization 0.5

# setup LLM
# llm = VllmLLM(n=1)

# local coati api
model_path = os.environ.get('LLAMA2_PATH')
coati_api = CoatiAPI('llama', model_path)
llm = CoatiLLM(n=1, api=coati_api)

# define the retriever
information_retriever = CustomRetriever()

# setup embedding model locally
embedding = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base",
                           model_kwargs={'device': 'cpu'},encode_kwargs={'normalize_embeddings': False})

# define memory with summarization ability
memory = ConversationBufferWithSummary(llm=llm,
        llm_kwargs={'max_new_tokens':50, 'temperature':0.6, 'do_sample':True})

# define the chain to preprocess the input
# disambiguate the input. e.g. "What is the capital of that country?" -> "What is the capital of France?"
llm_chain_disambiguate = LLMChain(llm=llm, prompt=PROMPT_DISAMBIGUATE_EN, llm_kwargs={'max_new_tokens':30, 'temperature':0.6, 'do_sample':True})

def disambiguity(input):
    out = llm_chain_disambiguate.run(input=input, chat_history=memory.buffer, stop=['\n'])
    return out.split('\n')[0] 

if __name__ == '__main__':
    # Load data to vector store
    print("Select files for constructing retriever")
    documents = []
    while True:
        file = input("Select a file to load or enter Esc to exit:")
        if file=='Esc':
            break
        data_name = input("Enter a short description of the data:")
        retriever_data = DocumentLoader([[file, data_name.replace(' ', '_')]]).all_data

        # Split
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        text_splitter = NeuralTextSplitter()
        splits = text_splitter.split_documents(retriever_data)
        documents.extend(splits)
    # create vector store
    vectordb = Chroma.from_documents(documents=documents, embedding=embedding)
    # initiate retriever    
    retriever=vectordb.as_retriever(search_kwargs={"k":3})
    information_retriever.set_retriever(retriever=retriever)
    information_retriever.set_k(k=3)

    # set document retrieval chain, we need this chain to calculate prompt length
    memory.initiate_document_retrieval_chain(llm, PROMPT_RETRIEVAL_QA_EN, information_retriever, 
        chain_type_kwargs={'chat_history':'', }, max_tokens=2000, input_key='question', memory_key='chat_history')

    # define retrieval chain
    retrieval_chain = RetrievalQA.from_chain_type(llm=llm, verbose=False, chain_type="stuff", retriever=information_retriever, 
                                            chain_type_kwargs={"prompt": PROMPT_RETRIEVAL_QA_EN,"memory":memory },
                                            llm_kwargs={'max_new_tokens':50, 'temperature':0.75, 'do_sample':True})
    # set disambiguity handler
    information_retriever.set_rephrase_handler(disambiguity)

    # start conversation
    while True:
        user_input = input("User: ")
        print(f"User: {user_input}")
        if 'END' == user_input:
            print("Agent: Happy to chat with you ：)")
            break    
        agent_response = retrieval_chain.run(query=user_input, stop = ['Human: '])
        agent_response = agent_response.split('\n')[0]
        print(f"Agent: {agent_response}")
     