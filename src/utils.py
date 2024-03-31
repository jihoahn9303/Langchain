from typing import List

import streamlit as st
from langchain.schema import Document


def save_message(message, role) -> None:
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)
        
def visualize_chat_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"], 
            message["role"], 
            save=False
        )

def format_documents(documents: List[Document]) -> str:
    return "\n\n".join(
        document.page_content
        for document in documents
    )