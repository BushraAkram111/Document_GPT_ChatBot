
import os
import streamlit as st
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.vectorstores import FAISS
import pandas as pd
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your file")
    st.header("DocumentGPT")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'csv'], accept_multiple_files=True)
        google_api_key = st.secrets["AIzaSyCis3PQiQJBzd1p58NRGSUq_E5-SKLoLs8"]
        process = st.button("Process")
        if process:
            if not google_api_key:
                st.info("Please add your Google API key to continue.")
                st.stop()
            files_text = get_files_text(uploaded_files)
            st.write("File loaded...")
            text_chunks = get_text_chunks(files_text)
            st.write("File chunks created...")
            vectorstore = get_vectorstore(text_chunks)
            st.write("Vector Store Created...")
            st.session_state.conversation = vectorstore
            st.session_state.processComplete = True
            st.session_state.session_id = os.urandom(16).hex()  # Initialize a unique session ID

    if st.session_state.processComplete:
        input_query = st.chat_input("Ask Question about your files.")
        if input_query:
            response_text = rag(st.session_state.conversation, input_query, google_api_key)
            st.session_state.chat_history.append({"content": input_query, "is_user": True})
            st.session_state.chat_history.append({"content": response_text, "is_user": False})

            response_container = st.container()
            with response_container:
                for i, message_data in enumerate(st.session_state.chat_history):
                    message(message_data["content"], is_user=message_data["is_user"], key=str(i))


def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        elif file_extension == ".csv":
            text += get_csv_text(uploaded_file)
    return text

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def get_csv_text(file):
    df = pd.read_csv(file)
    return df.to_string()

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)

def rag(vector_db, input_query, google_api_key):
    try:
        template = """You are an AI assistant that assists users by providing answers to their questions by extracting information from the provided context:
        {context}.
        If you do not find any relevant information from context for the given question, simply say 'I do not know'. Do not try to make up an answer.
        Answer should not be greater than 5 lines.
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()})

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)
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

if __name__ == '__main__':
    main()
