import os
import re

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.logging_config import logger


class DocumentProcessor:

    def __init__(self, model_encoding: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(model_encoding)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            length_function=self.calculate_tokens,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def clean_text(self, text: str) -> str:
        """Remove excessive spaces while maintaining the document's structure."""

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def calculate_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def process_document(self, documents: list[Document]) -> list[Document]:
        """Cleaning, chunking and enchanced metadata."""
        processed_chunks = []

        for document in documents:

            cleaned_text = self.clean_text(document.page_content)

            chunks = self.splitter.create_documents(
                [cleaned_text],
                metadatas=[document.metadata]
            )

            processed_chunks.extend(chunks)

        for i, chunk in enumerate(processed_chunks):

            filename = os.path.basename(chunk.metadata["source"])

            filename_no_extension = os.path.splitext(filename)[0]

            source, category = filename_no_extension.split(
                ",",
                maxsplit=1
            )

            chunk.metadata["source"] = source.strip()

            chunk.metadata["page"] = int(chunk.metadata["page"]) + 1
            
            chunk.metadata["category"] = category.strip()

            chunk.metadata["chunk_id"] = int(i)

            token_count = self.calculate_tokens(
                chunk.page_content
            )

            logger.info(
                f"Chunk {i} created: {token_count} tokens "
                f"(page {chunk.metadata['page']})."
            )

        return processed_chunks