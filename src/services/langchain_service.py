"""LangChain service for text processing and embeddings."""

from typing import List, Optional
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from src.config import config_manager
from src.db.chroma import get_chroma_path

class LangChainService:
    """Service for LangChain operations including chunking and embeddings."""

    def __init__(self):
        self.config = config_manager.config
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self._vectorstore = None

    def get_vectorstore(self) -> Chroma:
        """Get or create the Chroma vectorstore."""
        if self._vectorstore is None:
            chroma_path = get_chroma_path()
            self._vectorstore = Chroma(
                persist_directory=str(chroma_path),
                embedding_function=self.embeddings,
                collection_name="elumine_artifacts"
            )
        return self._vectorstore

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using LangChain's text splitter."""
        if not text.strip():
            return []

        chunks = self.text_splitter.split_text(text)
        return chunks

    def create_documents(
        self,
        text: str,
        metadata: dict,
        filename: str,
        batch_id: int
    ) -> List[Document]:
        """Create LangChain documents from text with metadata."""
        chunks = self.chunk_text(text)

        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "filename": filename,
                "batch_id": batch_id,
                "chunk_id": f"artifact_{batch_id}_{Path(filename).stem}_chunk_{i}"
            }
            documents.append(Document(page_content=chunk, metadata=doc_metadata))

        return documents

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vectorstore and return their IDs."""
        vectorstore = self.get_vectorstore()

        # Generate IDs from metadata
        ids = [doc.metadata.get("chunk_id") for doc in documents]

        # Add documents with embeddings
        vectorstore.add_documents(documents, ids=ids)

        return ids

    def search_similar(self, query: str, k: int = 5, filter_dict: Optional[dict] = None) -> List[Document]:
        """Search for similar documents in the vectorstore."""
        vectorstore = self.get_vectorstore()

        if filter_dict:
            return vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            return vectorstore.similarity_search(query, k=k)

    def search_by_batch(self, query: str, batch_id: int, k: int = 5) -> List[Document]:
        """Search for documents within a specific batch."""
        return self.search_similar(query, k=k, filter_dict={"batch_id": batch_id})

    def search_by_filename(self, query: str, filename: str, k: int = 5) -> List[Document]:
        """Search for documents from a specific filename."""
        return self.search_similar(query, k=k, filter_dict={"filename": filename})

# Singleton instance
_langchain_service = None

def get_langchain_service() -> LangChainService:
    """Get the singleton LangChain service instance."""
    global _langchain_service
    if _langchain_service is None:
        _langchain_service = LangChainService()
    return _langchain_service
