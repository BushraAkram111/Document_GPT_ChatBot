import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client.http.exceptions import ResponseHandlingException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import Docx2txtLoader
from dotenv import load_dotenv
import tempfile

# Load environment variables from .env file
load_dotenv()

# Default API keys
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
DEFAULT_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

# Qdrant credentials
QDRANT_API_KEY = "-H67duistzh3LrcFwG4eL2-M_OLvlj-D2czHgEdvcOYByAn5BEP5kA"
QDRANT_URL = "https://11955c89-e55c-47df-b9dc-67a3458f2e54.us-east4-0.gcp.cloud.qdrant.io"

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

# Function to get files and load their text
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

# Function to get text chunks from the documents
def get_text_chunks(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    for page in pages:
        text = page.page_content
        chunks = text_splitter.split_text(text)
        texts.extend(chunks)
    return texts

# Function to get the vector store
def get_vectorstore(text_chunks, qdrant_api_key, qdrant_url):
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Qdrant.from_texts(text_chunks, embeddings_model, api_key=qdrant_api_key, url=qdrant_url)
    return vectorstore

# Function to perform Retrieval-Augmented Generation (RAG)
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
        context_docs = retriever.get_relevant_documents(input_query)
        context_text = "\n".join([doc.page_content for doc in context_docs])

        if selected_model == "OpenAI":
            chat_model = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
        else:
            chat_model = ChatGoogleGenerativeAI(api_key=google_api_key)

        # Create the chat prompt
        messages = [HumanMessage(content=input_query), AIMessage(content=context_text)]
        response = chat_model(messages)

        return response['content']
    except Exception as e:
        return f"An error occurred: {e}"

# Main function to run the Streamlit app
def main():
    st.markdown("<h1 style='text-align: center; color: #0073e6;'>Elevate Your Document Experience with RAG GPT and Conversational AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #0073e6;'>🤖 Choose Your AI Model: Select from OpenAI or Google Gemini for tailored responses.</h3>", unsafe_allow_html=True)

    # File uploader at the front
    uploaded_files = st.file_uploader("🔍 Upload Your Files", type=['pdf', 'docx', 'csv', 'txt'], accept_multiple_files=True, label_visibility="visible")

    if uploaded_files:
        st.sidebar.header("Model Selection")
        model_choice = st.sidebar.radio("Select the model to use", ("Google Gemini", "OpenAI"))
        st.session_state.selected_model = model_choice

        process = st.sidebar.button("Process")
        if process:
            st.session_state.qdrant_api_key = QDRANT_API_KEY
            st.session_state.qdrant_url = QDRANT_URL

            pages = get_files_text(uploaded_files)
            if pages:
                st.sidebar.write(f"Total pages loaded: {len(pages)}")
                text_chunks = get_text_chunks(pages)
                st.sidebar.write(f"File chunks created: {len(text_chunks)}")
                if text_chunks:
                    try:
                        vectorstore = get_vectorstore(text_chunks, QDRANT_API_KEY, QDRANT_URL)
                        st.sidebar.write("Vector Store Created...")
                        st.session_state.conversation = vectorstore
                        st.session_state.processComplete = True
                        st.session_state.session_id = os.urandom(16).hex()  # Initialize a unique session ID
                        st.success("Processing complete! You can now ask questions about your document.")
                    except ResponseHandlingException as e:
                        st.error(f"Qdrant API Error: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

    if st.session_state.processComplete:
        st.header("💬 Chat with Your Document")
        user_input = st.text_input("Ask a question about your document:", key="input_query")

        if user_input:
            with st.spinner("Generating response..."):
                answer = rag(st.session_state.conversation, user_input, DEFAULT_OPENAI_API_KEY, DEFAULT_GOOGLE_API_KEY, st.session_state.selected_model)
                st.write("**Answer:**", answer)

                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append((user_input, answer))

        if st.session_state.chat_history:
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
                st.write(f"**You:** {user_msg}", unsafe_allow_html=True)
                st.write(f"**Bot:** {bot_msg}", unsafe_allow_html=True)

if __name__ == "__main__":
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processComplete' not in st.session_state:
        st.session_state.processComplete = False
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "OpenAI"

    main()
