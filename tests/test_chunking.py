from langchain_core.documents import Document

from app.chunking import DocumentProcessor


def test_clean_text():
    processor = DocumentProcessor()

    text = "Hola    mundo\n\n\n\n  esto   es una prueba"

    result = processor.clean_text(text)

    assert result == "Hola mundo\n\nesto es una prueba"


def test_calculate_tokens():
    processor = DocumentProcessor()

    text = "The correlation study the relationship between two variables."

    result = processor.calculate_tokens(text)

    assert isinstance(result, int)
    assert result > 0
    assert result == len(processor.tokenizer.encode(text))


def test_process_document_crea_chunks():
    processor = DocumentProcessor()

    document = Document(
        page_content=(
            "The regression is a statistical technique used to make "
            "predictions. Correlation studies the relationship between two variables."
        ),
        metadata={
            "source": "Estadística,correlacion.pdf",
            "page": 0,
        },
    )

    chunks = processor.process_document([document])

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_process_document_add_metadata():
    processor = DocumentProcessor()

    document = Document(
        page_content="The correlation study the relationship between two variables.",
        metadata={
            "source": "Estadística,correlacion.pdf",
            "page": 0,
        },
    )

    chunks = processor.process_document([document])

    assert len(chunks) > 0

    metadata = chunks[0].metadata

    assert metadata["source"] == "Estadística"
    assert metadata["category"] == "correlacion"
    assert metadata["page"] == 1
    assert metadata["chunk_id"] == 0
    assert isinstance(metadata["chunk_id"], int)