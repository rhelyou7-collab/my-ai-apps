import os
import tempfile
import streamlit as st

# Modern LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# CORRECTED CHAIN IMPORTS
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(
    page_title="Local PDF Chatbot", page_icon="🤖", layout="centered"
)
st.title("📁 Local PDF Chatbot (100% Offline)")
st.caption(
    "Running locally via Ollama (Llama 3.2:3b) & ChromaDB optimized for 6 GB RAM."
)

# Initialize Session State
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file and st.session_state.rag_chain is None:
        with st.spinner("Processing PDF and building local vector index..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # 1. Load PDF
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()

                # 2. Split chunks (optimized size for small RAM)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=50
                )
                texts = text_splitter.split_documents(documents)

                # 3. Use Ollama Embeddings to save RAM (avoids loading PyTorch)
                #embeddings = OllamaEmbeddings(model="llama3.2:3b")
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                
                # 4. Build local vector store
                vectorstore = Chroma.from_documents(texts, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                # 5. Connect to ChatOllama with strict parameters
                llm = ChatOllama(
                    model="llama3.2:3b",
                    temperature=0.0,      # Eliminates hallucinations
                    num_ctx=4096          # Reduced context window to save RAM (was 8192)
                )

                # 6. Multi-turn Chat Context Manager
                contextualize_q_system_prompt = (
                    "Given a chat history and the latest user question "
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do NOT answer the question, just reformulate it if needed."
                )
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, contextualize_q_prompt
                )

                # 7. Enforce Strict Document Prompt Boundaries
                qa_system_prompt = (
                    "You are a helpful assistant for question-answering tasks.\n"
                    "Use the following pieces of retrieved context to answer the question.\n"
                    "If you don't know the answer, say exactly: 'I cannot find that in the document.'\n"
                    "Do NOT make up information outside this context.\n\n"
                    "<CONTEXT_START>\n"
                    "{context}\n"
                    "<CONTEXT_END>"
                )
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                
                # Combine into modern Retrieval Chain
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                st.session_state.rag_chain = create_retrieval_chain(
                    history_aware_retriever, question_answer_chain
                )

                st.success("PDF processed successfully! You can now chat below.")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    if st.session_state.rag_chain is not None:
        if st.button("Clear Document / Reset"):
            st.session_state.rag_chain = None
            st.session_state.messages = []
            st.rerun()

# Main Chat Interface
if st.session_state.rag_chain is None:
    st.info("👈 Please upload a PDF file in the sidebar to start chatting with it.")
else:
    # Display chat history from UI memory
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_query := st.chat_input("Ask a question about your PDF..."):
        # Display user message instantly
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate streaming response from local model
        with st.chat_message("assistant"):
            try:
                # Convert Streamlit history format to proper LangChain Message objects
                formatted_history = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        formatted_history.append(HumanMessage(content=msg["content"]))
                    else:
                        formatted_history.append(AIMessage(content=msg["content"]))

                # Helper generator function to stream tokens chunk by chunk
                def response_generator():
                    for chunk in st.session_state.rag_chain.stream({
                        "input": user_query, 
                        "chat_history": formatted_history
                    }):
                        if "answer" in chunk:
                            yield chunk["answer"]

                answer = st.write_stream(response_generator())
                
                # Store text variables inside session state for history persistence
                st.session_state.messages.append({"role": "user", "content": user_query})
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"Error generating response. Details: {e}"
                st.error(error_msg)
