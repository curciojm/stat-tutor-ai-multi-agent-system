from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document

from app.chunking import DocumentProcessor
from app.db_config import NAMESPACE


def pdf_processing():
    print("📄 Pinecone empty. Processing PDFs...")

    loader = DirectoryLoader(
        "data",
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    raw_documments = loader.load()

    processor = DocumentProcessor()

    return processor.process_document(raw_documments)

def retrieve_pinecone_documents(index):
    print("📦 Retrieving documents from pinecone...")

    ids = []

    for page in index.list(namespace=NAMESPACE):
        ids.extend(page)

    documents = []

    batch_size = 100

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]

        fetched_vectors = index.fetch(
            ids=batch_ids,
            namespace=NAMESPACE
        )

        for vector in fetched_vectors.vectors.values():
            metadata = vector.metadata

            documents.append(
                Document(
                    page_content=metadata["text"],
                    metadata=metadata
                )
            )

    print(f"📄 Retrieved documents: {len(documents)}")

    return documents