from typing import Union

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit.delta_generator import DeltaGenerator
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores.chroma import Chroma
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory

from src.utils import save_message


# @st.cache_data
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file: UploadedFile) -> VectorStoreRetriever:
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    
    with open(file_path, "wb") as f:
        f.write(file_content)
        
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=embeddings,
        document_embedding_cache=cache_dir
    )
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=cached_embeddings
    )
    retriever = vectorstore.as_retriever()
    
    return retriever

@st.cache_resource
def initialize_llm(mem_type: Union[str] = None) -> Union[BaseChatModel]:
    if mem_type == 'chat':
        
        class ChatCallbackHandler(BaseCallbackHandler):
            message = ""
            
            def on_llm_start(self, *args, **kwargs):
                self.message_box: DeltaGenerator = st.empty()
                
            def on_llm_end(self, *args, **kwargs):
                save_message(self.message, "ai")
                self.message = ""
            
            def on_llm_new_token(self, token: str, *args, **kwargs):
                self.message += token
                self.message_box.markdown(self.message)
                
        callbacks = [ChatCallbackHandler()]
    
    else:  # mem_type == memory
        callbacks = []
    
    return ChatOpenAI(
        temperature=0.2,
        streaming=True,
        callbacks=callbacks
    )   
    
@st.cache_resource
def initialize_memory(_llm: BaseChatModel) -> ConversationSummaryBufferMemory:
    return ConversationSummaryBufferMemory(
        llm=_llm,
        return_messages=True,
        max_token_limit=200,
        memory_key='chat_history'
    )