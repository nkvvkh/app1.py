import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os

# Set up Google API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Load and extract text from PDF
def extract_text_from_pdf(https://arxiv.org/pdf/2505.06633):
    pdf_reader = PdfReader(https://arxiv.org/pdf/2505.06633)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Split text into documents
def split_text_to_docs(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents([Document(page_content=text)])

# Create a vector store from the document chunks
@st.cache_resource
def create_vector_store(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(docs, embeddings)

# RAG response generation
def generate_response(query, vector_store):
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.2
    )

    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"""Use the following context from a research paper to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

    response = model.invoke(prompt)
    return response.content

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RAG Chatbot with Gemini", page_icon="🧠")
st.title("📚 RAG Chatbot (Google Gemini + arXiv PDF)")
st.write("Upload a research paper PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        docs = split_text_to_docs(text)
        vector_store = create_vector_store(docs)

    query = st.text_input("Ask a question about the paper:")

    if query:
        with st.spinner("Generating answer..."):
            answer = generate_response(query, vector_store)
        st.subheader("Answer")
        st.write(answer)
