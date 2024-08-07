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
        /* Updated CSS for text colors and responsive design */
        .text-blue {
            color: #0073e6;
        }
        /* Responsive design */
        @media (max-width: 768px) {
            .main {
                padding: 10px;
            }
            .sidebar .sidebar-content {
                padding: 10px;
            }
            .stButton button {
                padding: 8px 16px;
            }
            .message {
                padding: 10px;
            }
            .chat-input {
                padding: 8px;
            }
        }
        @media (max-width: 480px) {
            .text-blue {
                font-size: 1.5em;
            }
            h1.text-blue {
                font-size: 1.5em;
            }
            h3.text-blue {
                font-size: 1.2em;
            }
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
DEFAULT_OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
DEFAULT_GOOGLE_API_KEY = "AIzaSyCis3PQiQJBzd1p58NRGSUgyu7owwww5g7gbq_E5-SKLoLs8"

# Qdrant credentials (hidden from users)
QDRANT_API_KEY = "-H67duistzh3LrcFwG4eL2-M_OLvlj-D2czHgEdvcOYByAn5BEP5kA"
QDRANT_URL = "https://11955c89-e55c-47df-b9dc-67a3458f2e54.us-east4-0.gcp.cloud.qdrant.io"

def main():
    load_dotenv()

    st.markdown("<h1 class='text-blue' style='text-align: center;'>Chat with Documents</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='text-blue' style='text-align: center;'>🤖 Choose Your AI Model: Select from OpenAI or Google Gemini for tailored responses.</h3>", unsafe_allow_html=True)

    # File uploader at the front
    uploaded_files = st.file_uploader("🔍 Upload Your Files", type=['pdf', 'docx', 'csv', 'txt'], accept_multiple_files=True, label_visibility="visible")

    if uploaded_files:
        st.sidebar.header("Model Selection")
        model_choice = st.sidebar.radio("Select the model to use", ("Google Gemini", "OpenAI"))
        st.session_state.selected_model = model_choice

        st.sidebar.write("### API Keys")
        openai_api_key = DEFAULT_OPENAI_API_KEY
        google_api_key = AIzaSyBTfA6_lri8MtjYKccTMZ8umT_uvXa6hHU

        st.session_state.openai_api_key = openai_api_key
        st.session_state.google_api_key = google_api_key

        st.session_state.qdrant_api_key = QDRANT_API_KEY
        st.session_state.qdrant_url = QDRANT_URL

        st.sidebar.write("### Process Files")
        process = st.sidebar.button("Process")
        if process:
            pages = get_files_text(uploaded_files)
            if pages:
                st.sidebar.write(f"Total pages loaded: {len(pages)}")
                text_chunks = get_text_chunks(pages)
                st.sidebar.write(f"File chunks created: {len(text_chunks)} chunks")
                if text_chunks:
                    vectorstore = get_vectorstore(text_chunks, QDRANT_API_KEY, QDRANT_URL)
                    st.sidebar.write("Vector Store Created...")
                    st.session_state.conversation = vectorstore
                    st.session_state.processComplete = True
                    st.session_state.session_id = os.urandom(16).hex()  # Initialize a unique session ID
                    st.success("Processing complete! You can now ask questions about your files.")
                else:
                    st.error("Failed to create text chunks.")
            else:
                st.error("No pages loaded from files.")

    if st.session_state.processComplete:
        st.subheader("Chat with Your Document")
        input_query = st.text_input("Ask a question about your files:", key="chat_input")

        if input_query:
            response_text = rag(st.session_state.conversation, input_query, st.session_state.openai_api_key, st.session_state.google_api_key, st.session_state.selected_model)
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
    vectorstore = Qdrant.from_texts(text_chunks, embeddings_model, api_key=qdrant_api_key, url=qdrant_url)
    return vectorstore

def get_text_chunks(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    for doc in pages:
        chunks = text_splitter.split_text(doc.page_content)
        texts.extend(chunks)
    return texts

def rag(vector_db, input_query, openai_api_key, google_api_key, selected_model):
    try:
        template = """
        You are a helpful assistant. Provide accurate and relevant answers based on the context of the documents uploaded.
        If you do not know the answer, you should say "I don't know."
        Context: {context}
        Question: {question}
        """

        context = vector_db.similarity_search(input_query)
        context_text = " ".join([c.page_content for c in context])
        prompt = template.format(context=context_text, question=input_query)

        if selected_model == "OpenAI":
            model = ChatOpenAI(openai_api_key=openai_api_key)
            response = model([HumanMessage(content=prompt), AIMessage(content="")])
            response_text = response['text']  # Access the response text correctly

        elif selected_model == "Google Gemini":
            model = ChatGoogleGenerativeAI(api_key=google_api_key)
            response = model([HumanMessage(content=prompt), AIMessage(content="")])
            response_text = response['text']  # Access the response text correctly

        else:
            response_text = "Invalid model selected."

        return response_text

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processComplete' not in st.session_state:
        st.session_state.processComplete = False

    main()
