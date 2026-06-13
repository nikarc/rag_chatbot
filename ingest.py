import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Config
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_chatbot"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents():
    """Load PDF documents from the data/ directory"""
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {DATA_DIR}/")
    return documents


def split_documents(documents):
    """Split documents in to chunks optimized for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(
        f"Split into {len(chunks)} chunks "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return chunks


def create_vector_store(chunks):
    """Create embeddings and store them in ChromaDB"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(
        f"Created vector store with {vector_store._collection.count()} "
        f"embeddings in {CHROMA_DIR}/"
    )
    return vector_store


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_store(chunks)
