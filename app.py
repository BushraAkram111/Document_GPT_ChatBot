import os
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
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

# Set Google Gemini API Key
GOOGLE_API_KEY = "AIzaSyCis3PQiQJBzd1p58NRGSUq_E5-SKLoLs8"

def main():
    load_dotenv()

    st.markdown("<h1 style='text-align: center; color: #0073e6;'>Elevate Your Document Experience with RAG GPT and Conversational AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #0073e6;'>🤖 Choose Your AI Model: Select from OpenAI or Google Gemini for tailored responses.</h3>", unsafe_allow_html=True)

    if 'api_key_error' in st.session_state and st.session_state.api_key_error:
        st.sidebar.error("Please enter all required API keys.")

    # File uploader at the front
    uploaded_files = st.file_uploader("🔍 Upload Your Files", type=['pdf', 'docx', 'csv', 'txt'], accept_multiple_files=True, label_visibility="visible")

    if uploaded_files:
        st.sidebar.header("Model Selection and API Keys")
        model_choice = st.sidebar.radio("Select the model to use", ("Google Gemini", "OpenAI"))
        st.session_state.selected_model = model_choice

        qdrant_api_key = st.sidebar.text_input("Qdrant API Key", type="password", help="Enter your Qdrant API Key here.")
        qdrant_url = st.sidebar.text_input("Qdrant URL", help="Enter your Qdrant URL here.")
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API Key here.")

        # Check for missing API keys
        if qdrant_api_key and qdrant_url and openai_api_key:
            st.session_state.qdrant_api_key = qdrant_api_key
            st.session_state.qdrant_url = qdrant_url
            st.session_state.openai_api_key = openai_api_key

            # Clear any previous API key error
            st.session_state.api_key_error = False
        else:
            st.session_state.api_key_error = True

        if not st.session_state.api_key_error:
            process = st.sidebar.button("Process")
            if process:
                pages = get_files_text(uploaded_files)
                if pages:
                    st.sidebar.write(f"Total pages loaded: {len(pages)}")
                    text_chunks = get_text_chunks(pages)
                    st.sidebar.write(f"File chunks created: {len(text_chunks)} chunks")
                    if text_chunks:
                        vectorstore = get_vectorstore(text_chunks, qdrant_api_key, qdrant_url)
                        st.sidebar.write("Vector Store Created...")
                        st.session_state.conversation = vectorstore
                        st.session_state.processComplete = True
                        st.session_state.session_id = os.urandom(16).hex()  # Initialize a unique session ID
                        st.success("Processing complete! You can now ask questions about your files.")
                    else:
                        st.error("Failed to create text chunks.")
                else:
                    st.error("No pages loaded from files.")
        else:
            st.error("Please enter all required API keys.")

    if st.session_state.processComplete:
        st.subheader("Chat with Your Document")
        input_query = st.text_input("Ask Question about your files.", key="chat_input")
        if input_query:
            response_text = rag(st.session_state.conversation, input_query, st.session_state.openai_api_key, GOOGLE_API_KEY, st.session_state.selected_model)
            st.session_state.chat_history.append({"content": input_query, "is_user": True})
            st.session_state.chat_history.append({"content": response_text, "is_user": False})

        response_container = st.container()
        with response_container:
            for i, message_data in enumerate(st.session_state.chat_history):
                message(message_data["content"], is_user=message_data["is_user"], key=str(i))

def get_files_text(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        if file_extension == ".pdf":
            loader = PyMuPDFLoader(temp_file_path)
            pages = loader.load()
        elif file_extension == ".csv":
            loader = CSVLoader(file_path=temp_file_path)
            pages = loader.load()
        elif file_extension == ".docx":
            loader = Docx2txtLoader(temp_file_path)
            pages = loader.load()
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)
            pages = loader.load()
        else:
            st.error("Unsupported file format.")
            return []

        documents.extend(pages)

        # Remove the temporary file
        os.remove(temp_file_path)
        
    return documents

def get_vectorstore(text_chunks, qdrant_api_key, qdrant_url):
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Qdrant.from_texts(texts=text_chunks, embedding=embeddings_model, collection_name="Machine_learning", url=qdrant_url, api_key=qdrant_api_key, force_recreate=True)
    return vectorstore

def get_text_chunks(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    texts = []
    for page in pages:
        text = page.page_content
        chunks = text_splitter.split_text(text)
        texts.extend(chunks)
    return texts

def rag(vector_db, input_query, openai_api_key, google_api_key, selected_model):
    try:
        template = """
        You are a helpful assistant. You will help the user by providing relevant answers to their questions based on the provided context. If you do not know the answer, just say that you don't know. Offer as much relevant information as possible in your response.

        Question: {question}

        Context: {context}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )

        if selected_model == "Google Gemini":
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=google_api_key)
        elif selected_model == "OpenAI":
            model = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
        else:
            raise ValueError("Invalid model selected.")
        output_parser = StrOutputParser()
        rag_chain = (
            setup_and_retrieval
            | prompt
            | model
            | output_parser
        )
        response = rag_chain.invoke(input_query)
        return response
    except Exception as ex:
        return str(ex)

if __name__ == "__main__":
    if 'processComplete' not in st.session_state:
        st.session_state.processComplete = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'google_api_key' not in st.session_state:
        st.session_state.google_api_key = GOOGLE_API_KEY
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = None
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = None
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None

    main()
