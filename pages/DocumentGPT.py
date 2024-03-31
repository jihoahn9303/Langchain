from typing import List

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

from src.cache_resource import (
    embed_file,
    initialize_llm,
    initialize_memory
)
from src.utils import (
    send_message,
    visualize_chat_history,
    format_documents
)


st.set_page_config(
    page_title="SmartDoc",
    page_icon="📄"
)

    
def load_memory(input) -> List[str]:
    return memory.load_memory_variables({})["chat_history"]
    
prompt = ChatPromptTemplate.from_messages(
    messages=[
        ("system", 
         """
         Answer the question using ONLY the following context and chat history.
         If you don't know the answer, just say you don't know.
         Don't make anything up. 
              
         Context: {context}
         """),
        ("human", "-" * 30),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# initialize LLM
llm_for_chat = initialize_llm(mem_type='chat')
llm_for_memory = initialize_llm()

# initialize memory
memory = initialize_memory(_llm=llm_for_memory)

# setting web page(title, markdown, sidebar)
st.title("SmartDoc")

st.markdown(
    """
    Welcome!
    
    User this chatbot to ask questions to an AI about your files!
    
    Upload your files on the sidebar.
    """
)

# embed file uploader inside sidebar
with st.sidebar:
    file: UploadedFile = st.file_uploader(
        label="Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"]
    )

if file:
    retriever: VectorStoreRetriever = embed_file(file)
    
    # send_message("I'm ready! Ask away!", "ai", save=False)
    visualize_chat_history()
    
    message = st.chat_input("I'm ready! Ask away!")
    if message:
        send_message(message, "human")
        
        chain = {
            "context": retriever | RunnableLambda(func=format_documents),
            "chat_history": RunnableLambda(func=load_memory),
            "question": RunnablePassthrough()
        } | prompt | llm_for_chat
        
        with st.chat_message("ai"):
            response = chain.invoke(message).content
            memory.save_context(
                inputs={"input": message},
                outputs={"output": response}
            )
        
else:
    st.session_state["messages"] = [
        {"message": "Oh, your file is super-duper amazing!!", "role": "ai"}
    ]
    memory.clear()
    memory.save_context(
        {"input": "Oh, your file is super-duper amazing!!"},
        {"output": "Thank you, let me ask you a couple of questions!"}
    )
