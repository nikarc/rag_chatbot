import os
import json
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document
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


def load_json_documents():
    """Load JSON files from the data/ directory, flattening nested objects to text."""
    documents = []
    for path in glob.glob(os.path.join(DATA_DIR, "**/*.json"), recursive=True):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data):
                text = json.dumps(item, indent=2)
                documents.append(Document(page_content=text, metadata={"source": path, "index": i}))
        else:
            text = json.dumps(data, indent=2)
            documents.append(Document(page_content=text, metadata={"source": path}))
    return documents


def load_documents():
    """Load PDF and JSON documents from the data/ directory."""
    pdf_docs = PyPDFDirectoryLoader(DATA_DIR).load()
    json_docs = load_json_documents()
    documents = pdf_docs + json_docs
    print(f"Loaded {len(pdf_docs)} PDF pages and {len(json_docs)} JSON documents from {DATA_DIR}/")
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
