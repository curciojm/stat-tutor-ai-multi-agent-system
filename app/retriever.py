from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from app.db_ingest import processed_documents, vectorstore

bm25_retriever = BM25Retriever.from_documents(
    processed_documents
)

bm25_retriever.k = 5

vectorial_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5},
)

# Weights assigned after evaluation
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vectorial_retriever],
    weights=[0.25, 0.75],
)