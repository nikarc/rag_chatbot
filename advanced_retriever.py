from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def build_hybrid_retriever(chunks):
    """Create a hybrid retriever combining BM25 and vector search"""
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        chunks, embeddings, collection_name="hybrid_search"
    )

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6]
    )

    return ensemble
