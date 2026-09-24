import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Chat with PDF - AI & Groq", layout="centered")
st.title("📄 Chat with PDF using Groq Cloud")

# Sidebar Configuration
st.sidebar.header("Configuration")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

if groq_api_key and uploaded_file:
    # Check if a new file was uploaded or if we need to initialize session state
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing new PDF and building vector database..."):
            # Load PDF
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)

            # Use official LangChain HuggingFace embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            # Build vector store database and save to session state
            st.session_state.vectorstore = FAISS.from_documents(final_documents, embeddings)
            st.session_state.last_uploaded_file = uploaded_file.name
            
            # Reset chat messages when a new PDF is uploaded
            st.session_state.messages = []

    retriever = st.session_state.vectorstore.as_retriever()

    # Initialize active Groq model
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")

    # Setup LCEL RAG Chain
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Streamlit Chat Interface History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask something about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = rag_chain.invoke(user_query)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Please enter your Groq API key and upload a PDF in the sidebar to get started.")
