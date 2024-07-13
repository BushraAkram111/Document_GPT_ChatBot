from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Qdrant
from langchain_qdrant import Qdrant
from streamlit_chat import message
import streamlit as st
import os,tempfile
import qdrant_client

# Set page config at the beginning
st.set_page_config(page_title="Chat with your file", layout="wide")

google_api_key = "AIzaSyCis3PQiQJBzd1p58NRGSUq_E5-SKLoLs8"
api_key = "-H67duistzh3LrcFwG4eL2-M_OLvlj-D2czHgEdvcOYByAn5BEP5kA"
url = "https://11955c89-e55c-47df-b9dc-67a3458f2e54.us-east4-0.gcp.cloud.qdrant.io"

# Add CSS styles
st.markdown("""
    <style>
        .main {
            background-color:  #000000;
            padding: 20px;
            color:#ffffff;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
        }
        .sidebar .sidebar-content h2 {
            color: #333333;
            background-color: #f0f0f0;
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
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown("<h1 style='text-align: center; color: #0073e6;'>Document GPT</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #0073e6;'>Upload Your File and ask any Question from file</h3>", unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "OpenAI"

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'csv'], accept_multiple_files=True)

        process = st.button("Process")
        if process:
            pages = get_files_text(uploaded_files)
            st.write("File loaded...")
            if pages:
                st.write(f"Total pages loaded: {len(pages)}")
                text_chunks = get_text_chunks(pages)
                st.write(f"File chunks created: {len(text_chunks)} chunks")
                if text_chunks:
                    vectorstore = get_vectorstore(text_chunks)
                    st.write("Vector Store Created...")
                    st.session_state.conversation = vectorstore
                    st.session_state.processComplete = True
                    st.session_state.session_id = os.urandom(16).hex()  # Initialize a unique session ID
                else:
                    st.error("Failed to create text chunks.")
            else:
                st.error("No pages loaded from files.")

    if st.session_state.processComplete:
        input_query = st.chat_input("Ask Question about your files.")
        if input_query:
            response_text = rag(st.session_state.conversation, input_query)
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
            loader = PyPDFLoader(temp_file_path)
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

def get_embeddings():
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings_model
def get_vectorstore(text_chunks):
    vectorstore = Qdrant.from_texts(
        text_chunks,
        api_key = api_key,
        embedding=get_embeddings(),
        url=url,
        prefer_grpc=True,
        collection_name="my_documents",
        force_recreate=True)

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
        chunks = text_splitter.split_text(page.page_content)
        texts.extend(chunks)
    return texts

def _qdrant_client():
    client = qdrant_client.QdrantClient(
    url,
    api_key=api_key)
    vectorstore = Qdrant(
    client=client, collection_name="my_documents", 
    embeddings=get_embeddings())

    return vectorstore

vector_db = _qdrant_client()

def rag(vector_db, input_query):
    try:
        template = """
        You are a helpful assistant. Provide accurate and relevant answers based on the context of the documents uploaded.
        If you do not know the answer, you should say "I don't know."
        Context: {context}
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()})

        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3,google_api_key=google_api_key)         
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

main()
