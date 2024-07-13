import os
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import Docx2txtLoader
from dotenv import load_dotenv
import tempfile

# Set page config at the beginning
st.set_page_config(page_title="Chat with Your Document", layout="wide")

# Add CSS styles
st.markdown("""
    <style>
        .main {
            background-color:  #f0f0f0;
            padding: 20px;
            color: #000000;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
        }
        .sidebar .sidebar-content h2 {
            color: #333333;
            background-color: #ffffff;
        }
        .stButton button {
            background-color: #0073e6;
            color: #ffffff;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #005bb5;
        }
        .message {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        .message.user {
            background-color: #e6f7ff;
        }
        .message.bot {
            background-color: #f0f0f0;
        }
        .chat-input {
            background-color: #ffffff;
            border: 1px solid #d9d9d9;
            border-radius: 10px;
            padding: 10px;
            width: 100%;
            box-sizing: border-box;
        }
        .dark-mode .main {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .dark-mode .sidebar .sidebar-content {
            background-color: #2e2e2e;
            color: #ffffff;
        }
        .dark-mode .message {
            background-color: #2e2e2e;
            color: #ffffff;
        }
        .dark-mode .chat-input {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #444444;
        }
        .dark-mode .stButton button {
            background-color: #005bb5;
        }
        .dark-mode .stButton button:hover {
            background-color: #0073e6;
        }
    </style>
""", unsafe_allow_html=True)

# Function to check and set the mode
def set_mode():
    if st.session_state.get('dark_mode', False):
        st.markdown('<script>document.body.classList.add("dark-mode");</script>', unsafe_allow_html=True)
    else:
        st.session_state['dark_mode'] = False

set_mode()

# Set default values for API keys
DEFAULT_OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API Key
DEFAULT_GOOGLE_API_KEY = "AIzaSyCis3PQiQJBzd1p58NRGSUq_E5-SKLoLs8"

# Qdrant credentials (hidden from users)
QDRANT_API_KEY = "-H67duistzh3LrcFwG4eL2-M_OLvlj-D2czHgEdvcOYByAn5BEP5kA"
QDRANT_URL = "https://11955c89-e55c-47df-b9dc-67a3458f2e54.us-east4-0.gcp.cloud.qdrant.io"

def get_files_text(uploaded_files):
    pages = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                loader = PyMuPDFLoader(tmp_file.name)
                pages.extend(loader.load())
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
            pages.append(TextLoader(text))
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(uploaded_file)
            pages.extend(loader.load())
        elif uploaded_file.type == "text/csv":
            loader = CSVLoader(uploaded_file)
            pages.extend(loader.load())
    return pages

def get_text_chunks(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = []
    for page in pages:
        chunks = text_splitter.split_documents([page])
        text_chunks.extend(chunks)
    return text_chunks

def get_vectorstore(text_chunks, api_key, url):
    embeddings = HuggingFaceEmbeddings()
    qdrant_client = QdrantClient(url=url, api_key=api_key)
    vectorstore = Qdrant.from_documents(text_chunks, embeddings, client=qdrant_client)
    return vectorstore

def rag(conversation, query, openai_api_key, google_api_key, model_choice):
    response = ""
    if model_choice == "OpenAI":
        llm = ChatOpenAI(api_key=openai_api_key, temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{query}")
        ])
        response = llm(prompt.format_prompt(query=query))
    elif model_choice == "Google Gemini":
        client = ChatGoogleGenerativeAI(api_key=google_api_key, temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{query}")
        ])
        response = client(prompt.format_prompt(query=query))
    return response

def main():
    load_dotenv()

    st.markdown("<h1 style='text-align: center; color: #0073e6;'>Elevate Your Document Experience with RAG GPT and Conversational AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #0073e6;'>🤖 Choose Your AI Model: Select from OpenAI or Google Gemini for tailored responses.</h3>", unsafe_allow_html=True)

    # File uploader at the front
    uploaded_files = st.file_uploader("🔍 Upload Your Files", type=['pdf', 'docx', 'csv', 'txt'], accept_multiple_files=True, label_visibility="visible")

    if uploaded_files:
        st.sidebar.header("Model Selection and API Keys")
        model_choice = st.sidebar.radio("Select the model to use", ("Google Gemini", "OpenAI"))
        st.session_state.selected_model = model_choice

        # Get API keys from the user or use the default one
        st.sidebar.write("### Optional: Add Your API Keys")
        openai_api_key = st.sidebar.text_input("OpenAI API Key", value=DEFAULT_OPENAI_API_KEY if 'openai_api_key' not in st.session_state else st.session_state.openai_api_key)
        google_api_key = st.sidebar.text_input("Google API Key", value=DEFAULT_GOOGLE_API_KEY if 'google_api_key' not in st.session_state else st.session_state.google_api_key)
        st.session_state.openai_api_key = openai_api_key
        st.session_state.google_api_key = google_api_key

        if st.sidebar.button("Process"):
            if not openai_api_key and not google_api_key:
                st.error("Please add at least one API key to continue.")
            else:
                try:
                    st.spinner("Processing your document...")

                    # Process uploaded files
                    pages = get_files_text(uploaded_files)
                    text_chunks = get_text_chunks(pages)
                    vectorstore = get_vectorstore(text_chunks, QDRANT_API_KEY, QDRANT_URL)

                    st.success("Document processed successfully!")

                    # Add chat feature
                    st.markdown("### 💬 Ask Questions About Your Document")
                    st.session_state.chat_history = []

                    user_input = st.text_input("Type your question here:", "", key="chat_input")

                    if user_input:
                        model = st.session_state.selected_model
                        openai_api_key = st.session_state.openai_api_key
                        google_api_key = st.session_state.google_api_key

                        # Generate response based on model choice
                        if model == "OpenAI":
                            st.session_state.messages.append({"role": "user", "content": user_input})
                            llm = ChatOpenAI(api_key=openai_api_key, temperature=0.7)
                            prompt = ChatPromptTemplate.from_messages([
                                ("system", "You are a helpful assistant."),
                                ("user", "{query}")
                            ])
                            response = llm(prompt.format_prompt(query=user_input))
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        elif model == "Google Gemini":
                            st.session_state.messages.append({"role": "user", "content": user_input})
                            client = ChatGoogleGenerativeAI(api_key=google_api_key, temperature=0.7)
                            prompt = ChatPromptTemplate.from_messages([
                                ("system", "You are a helpful assistant."),
                                ("user", "{query}")
                            ])
                            response = client(prompt.format_prompt(query=user_input))
                            st.session_state.messages.append({"role": "assistant", "content": response})

                        # Show chat history
                        for i, message_data in enumerate(st.session_state.messages):
                            if message_data["role"] == "user":
                                message(message_data["content"], is_user=True, key=f"user_{i}")
                            else:
                                message(message_data["content"], key=f"bot_{i}")

if __name__ == "__main__":
    main()
