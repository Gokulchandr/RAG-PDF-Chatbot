import streamlit as st
import os
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(page_title="RAG AI Chatbot", layout="wide")
st.title("🤖 PDF RAG Chatbot")

# Ensure your token is entered here
HUGGINGFACE_API = "Your_API_Key" 
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API

@st.cache_resource
def get_vector_store(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)

# -----------------------------
# App Layout
# -----------------------------
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    vector_store = get_vector_store(uploaded_file)
    st.success("PDF Indexed!")
    user_input = st.text_input("Ask a question:")

    if user_input:
        with st.spinner("Generating answer..."):
            try:
                # SWITCHED MODEL: meta-llama/Llama-3.1-8B-Instruct is highly compatible
                llm = HuggingFaceEndpoint(
                    repo_id="meta-llama/Llama-3.1-8B-Instruct",
                    huggingfacehub_api_token=HUGGINGFACE_API,
                    temperature=0.1,
                    max_new_tokens=512,
                )

                # This wrapper handles the chat template and provider routing
                chat_model = ChatHuggingFace(llm=llm)

                retriever = vector_store.as_retriever(search_kwargs={"k": 3})

                # Simple Chat Prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant. Answer based ONLY on the context: {context}"),
                    ("human", "{question}")
                ])

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                # Chain
                chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | chat_model
                    | StrOutputParser()
                )

                response = chain.invoke(user_input)
                st.subheader("Answer:")
                st.markdown(response)

            except Exception as e:
                st.error(f"Execution Error: {e}")
                st.info("💡 Tip: If you get a 'Model not supported' error, try changing the repo_id to 'microsoft/Phi-3-mini-4k-instruct' which is also free and fast.")
else:
    st.info("Please upload a PDF to begin.")