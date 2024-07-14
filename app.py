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
            background-color: #f0f0f0; /* Light grey background */
            padding: 20px;
            color: #000000; /* Black text color */
        }
        .sidebar .sidebar-content {
            background-color: #ffffff; /* White background for sidebar */
            border-radius: 10px;
            padding: 20px;
        }
        .sidebar .sidebar-content h2 {
            color: #333333; /* Dark grey text color for sidebar headers */
            background-color: #ffffff; /* White background for sidebar headers */
        }
        .stButton button {
            background-color: #4CAF50; /* Green button background */
            color: #ffffff; /* White text color for buttons */
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #45a049; /* Darker green on hover */
        }
        .message {
            background-color: #ffffff; /* White background for messages */
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); /* Light shadow for messages */
        }
        .message.user {
            background-color: #e0f7fa; /* Light cyan background for user messages */
        }
        .message.bot {
            background-color: #f0f0f0; /* Light grey background for bot messages */
        }
        .chat-input {
            background-color: #ffffff; /* White background for input field */
            border: 1px solid #d9d9d9; /* Light grey border */
            border-radius: 10px;
            padding: 10px;
            width: 100%;
            box-sizing: border-box;
            color: #000000; /* Black text color */
        }
        .dark-mode .main {
            background-color: #2e2e2e; /* Dark grey background for dark mode */
            color: #ffffff; /* White text color for dark mode */
        }
        .dark-mode .sidebar .sidebar-content {
            background-color: #3c3c3c; /* Darker grey for sidebar in dark mode */
            color: #ffffff; /* White text color for sidebar in dark mode */
        }
        .dark-mode .message {
            background-color: #3c3c3c; /* Darker grey background for messages in dark mode */
            color: #ffffff; /* White text color for messages in dark mode */
        }
        .dark-mode .chat-input {
            background-color: #4a4a4a; /* Even darker grey for input field in dark mode */
            color: #ffffff; /* White text color for input field in dark mode */
            border: 1px solid #5a5a5a; /* Slightly lighter grey border for input field */
        }
        .dark-mode .stButton button {
            background-color: #5a5a5a; /* Dark grey background for buttons in dark mode */
        }
        .dark-mode .stButton button:hover {
            background-color: #6a6a6a; /* Slightly lighter grey on hover */
        }
        /* Custom styles for success messages */
        .stSuccess {
            background-color: #ffffff; /* White background for success messages */
            color: #000000; /* Black text color */
            border: 1px solid #000000; /* Black border */
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        /* Custom styles for the main heading */
        .main-heading {
            background-color: #ffffff; /* White background for the main heading */
            color: #000000; /* Black text color */
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        /* Custom styles for the chat input prompt */
        .chat-input-prompt {
            background-color: #ffffff; /* White background for input prompt */
            color: #000000; /* Black text color */
            border: 1px solid #d9d9d9; /* Light grey border */
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
        }
        /* Custom styles for subheader */
        .stSubheader {
            color: #000000; /* Black text color for subheader */
            padding: 10px;
            margin-bottom: 10px;
            background-color: #f0f0f0; /* Light grey background for subheader */
            border-radius: 5px;
        }
        /* Custom styles for text input */
        .stTextInput {
            background-color: #ffffff; /* White background for text input */
            color: #000000; /* Black text color for text input */
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
DEFAULT_GOOGLE_API_KEY = "AIzaSyCis3PQiQJBzd1p58NRGSUq_E5-SKLoLs8"

# Qdrant credentials (hidden from users)
QDRANT_API_KEY = "-H67duistzh3LrcFwG4eL2-M_OLvlj-D2czHgEdvcOYByAn5BEP5kA"
QDRANT_URL = "https://11955c89-e55c-47df-b9dc-67a3458f2e54.us-east4-0.gcp.cloud.qdrant.io"

def main():
    load_dotenv()

    st.markdown("<h1 class='main-heading'>Chat with Documents</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='main-heading'>🤖 Choose Your AI Model: Select from OpenAI or Google Gemini for tailored responses.</h3>", unsafe_allow_html=True)

    # File uploader at the front
    uploaded_files = st.file_uploader("🔍 Upload Your Files", type=['pdf', 'docx', 'csv', 'txt'], accept_multiple_files=True, label_visibility="visible")

    if uploaded_files:
        st.sidebar.header("Model Selection")
        model_choice = st.sidebar.radio("Select the model to use", ("Google Gemini", "OpenAI"))
        st.session_state.selected_model = model_choice

        st.sidebar.write("### API Keys")
        openai_api_key = DEFAULT_OPENAI_API_KEY
        google_api_key = DEFAULT_GOOGLE_API_KEY

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
        st.markdown("<div class='chat-input-prompt'>Ask a question about your files:</div>", unsafe_allow_html=True)
        input_query = st.text_input("", key="chat_input", placeholder="Type your question here...")

        if st.button("Submit"):
            if input_query:
                with st.spinner("Fetching response..."):
                    answer = rag(
                        st.session_state.conversation,
                        input_query,
                        st.session_state.openai_api_key,
                        st.session_state.google_api_key,
                        st.session_state.selected_model
                    )
                st.markdown(f"**Answer:** {answer}")

        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                message(chat['message'], is_user=chat['is_user'])

if __name__ == "__main__":
    main()
