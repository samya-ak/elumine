"""ChromaDB service for storing both vectors and metadata."""

import os
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from src.config import config_manager


def get_chroma_path() -> Path:
    """Get the path where ChromaDB data will be stored."""
    config = config_manager.config
    return Path(config.db_path) / "chroma_db"


class ChromaService:
    """Service for ChromaDB operations including vectors and metadata storage."""

    def __init__(self):
        self.config = config_manager.config

        # Set OpenAI API key from config if provided
        if self.config.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.config.openai_api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API key required. Set via config: "
                "elumine config --openai-api-key YOUR_KEY or set OPENAI_API_KEY env var. "
                "You can also set the LLM model: elumine config --llm-model gpt-4"
            )

        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=0
        )
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
                collection_name="elumine_data"
            )
        return self._vectorstore

    def create_batch(self, batch_name: Optional[str] = None) -> int:
        """Create a new batch and return its ID."""
        batch_id = int(time.time() * 1000)  # Use timestamp as unique ID

        # Create a dummy document for batch metadata
        batch_metadata = {
            "type": "batch",
            "batch_id": batch_id,
            "batch_name": batch_name or f"batch_{batch_id}",
            "created_at": datetime.now().isoformat(),
            "artifact_count": 0
        }

        batch_doc = Document(
            page_content=f"Batch: {batch_name or f'batch_{batch_id}'}",
            metadata=batch_metadata
        )

        vectorstore = self.get_vectorstore()
        vectorstore.add_documents([batch_doc], ids=[f"batch_{batch_id}"])

        return batch_id

    def create_documents(
        self,
        text: str,
        filename: str,
        batch_id: int,
        source_type: str = "text_file",
        source: str = "",
        filetype: str = "",
        **extra_metadata
    ) -> List[Document]:
        """Create LangChain documents from text with comprehensive metadata."""

        # Create base metadata that will be shared across all chunks
        base_metadata = {
            "type": "artifact_chunk",
            "batch_id": batch_id,
            "filename": filename,
            "source": source,
            "source_type": source_type,  # "youtube_video", "local_media", "text_file"
            "filetype": filetype,
            "created_at": datetime.now().isoformat(),
            "status": "processing",
            **extra_metadata
        }

        initial_doc = Document(page_content=text, metadata=base_metadata)

        # Use split_documents - preserves metadata automatically
        documents = self.text_splitter.split_documents([initial_doc])

        # Add chunk-specific metadata
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "chunk_index": i,
                "total_chunks": len(documents),
                "chunk_id": f"artifact_{batch_id}_{Path(filename).stem}_chunk_{i}",
                "artifact_id": f"artifact_{batch_id}_{Path(filename).stem}",
                "status": "ingested"
            })

        return documents

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vectorstore and return their IDs."""
        vectorstore = self.get_vectorstore()

        # Generate IDs from metadata
        ids = [doc.metadata.get("chunk_id") for doc in documents]

        # Add documents with embeddings
        vectorstore.add_documents(documents, ids=ids)

        return ids

    def _get_chunks(self, filter_dict: Optional[Dict[str, Any]] = None, k: int = 10000) -> List[Document]:
        """Internal method to get chunks with optional filtering."""
        vectorstore = self.get_vectorstore()

        # Build filter - always exclude batch documents
        base_filter = {"type": {"$eq": "artifact_chunk"}}

        if filter_dict:
            filters = [base_filter]
            for key, value in filter_dict.items():
                filters.append({key: {"$eq": value}})
            search_filter = {"$and": filters}
        else:
            search_filter = base_filter

        try:
            return vectorstore.similarity_search("", k=k, filter=search_filter)
        except Exception:
            # Fallback with simpler filter if complex filter fails
            try:
                results = vectorstore.similarity_search("", k=k, filter=filter_dict or {})
                return [r for r in results if r.metadata.get("type") == "artifact_chunk"]
            except Exception:
                return []

    def _group_chunks_by_artifact(self, chunks: List[Document]) -> Dict[str, Dict[str, Any]]:
        """Group chunks by artifact_id and return artifact metadata with chunk counts."""
        artifacts = {}
        for chunk in chunks:
            artifact_id = chunk.metadata.get("artifact_id")
            if artifact_id and artifact_id not in artifacts:
                metadata = chunk.metadata.copy()
                metadata["chunk_count"] = sum(1 for c in chunks if c.metadata.get("artifact_id") == artifact_id)
                artifacts[artifact_id] = metadata
        return artifacts

    def list_artifacts(self) -> List[Dict[str, Any]]:
        """List all artifacts (grouped by artifact_id)."""
        chunks = self._get_chunks()
        artifacts = self._group_chunks_by_artifact(chunks)
        return list(artifacts.values())

    def search_similar(self, query: str, k: int = 5, filter_dict: Optional[dict] = None) -> List[Document]:
        """Search for similar documents in the vectorstore."""
        vectorstore = self.get_vectorstore()

        # Build filter - always exclude batch documents
        base_filter = {"type": {"$eq": "artifact_chunk"}}

        if filter_dict:
            filters = [base_filter]
            for key, value in filter_dict.items():
                filters.append({key: {"$eq": value}})
            search_filter = {"$and": filters}
        else:
            search_filter = base_filter

        return vectorstore.similarity_search(query, k=k, filter=search_filter)

    def search_by_artifact(self, query: str, artifact_id: str, k: int = 5) -> List[Document]:
        """Search for documents within a specific artifact."""
        return self.search_similar(query, k=k, filter_dict={"artifact_id": artifact_id})

    def get_all_chunks_for_artifact(self, artifact_id: str) -> List[Document]:
        """Get all chunks for a specific artifact, sorted by chunk_index."""
        chunks = self._get_chunks({"artifact_id": artifact_id})
        chunks.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        return chunks

    def generate_answer(self, question: str, context_docs: List[Document]) -> str:
        """Generate an answer based on the question and context documents."""
        # Combine context from all documents
        context = "\n\n".join([doc.page_content for doc in context_docs])

        # Create prompt template
        template = """You are a helpful assistant that answers questions based on the provided context.
        Use only the information from the context to answer the question. If you cannot answer the question
        based on the context, say "I don't have enough information to answer that question."

        Context:
        {context}

        Question: {question}

        Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        # Create chain
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(question)

    def generate_summary(self, documents: List[Document]) -> str:
        """Generate a summary from the provided documents."""
        # Combine all document content
        content = "\n\n".join([doc.page_content for doc in documents])

        template = """Please provide a comprehensive summary of the following content.
        Focus on the main points, key insights, and important details.

        Content:
        {content}

        Summary:"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"content": lambda x: content}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke({})

    def generate_notes(self, documents: List[Document]) -> str:
        """Generate structured notes from the provided documents."""
        content = "\n\n".join([doc.page_content for doc in documents])

        template = """Please create structured notes from the following content.
        Organize the information with clear headings, bullet points, and key takeaways.
        Format the output in markdown.

        Content:
        {content}

        Structured Notes:"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"content": lambda x: content}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke({})


# Singleton instance
_chroma_service = None


def get_chroma_service() -> ChromaService:
    """Get the singleton ChromaDB service instance."""
    global _chroma_service
    if _chroma_service is None:
        _chroma_service = ChromaService()
    return _chroma_service
