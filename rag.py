
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load API key from .env in the project folder
load_dotenv(Path(__file__).resolve().parent / ".env")

PDF_PATH = Path(__file__).resolve().parent / "documents" / "GenAI_RAG_Sample.pdf"


def create_vector_store():

    # 1. Load PDF
    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    # 2. Split document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # 3. Get Gemini API key
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "API key not found. Check your .env file in the RAG folder."
        )

    # 4. Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    # 5. Store document embeddings in FAISS
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store
