import streamlit as st
from dotenv import load_dotenv
import os
from lanmgchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import time

#Load environment variables
load_dotenv()

# Set up Groq API key
#groq_api_key = os.getenv("GROQ_TITLE_API_KEY")

groq_api_key = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="Dynamic RAG with Groq", layout="wide")
st.image(
st.title("Dynamic RAG with Groq,FAISS, and Llama3")

#Initialize session state for vector store and chat history
if "vector" not in st.session_state:
    st.session_state.vector= None
if "chat_history" not in st.session_state:
  st.session_state.chat_history = []

# Sidebar for document upload
with st.sidebar:



