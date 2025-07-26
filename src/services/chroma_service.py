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

    def get_batch_info(self, batch_id: int) -> Optional[Dict[str, Any]]:
        """Get batch information by ID."""
        vectorstore = self.get_vectorstore()

        try:
            results = vectorstore.similarity_search(
                "",
                k=1,
                filter={"type": "batch", "batch_id": batch_id}
            )
            if results:
                return results[0].metadata
        except:
            pass
        return None

    def list_batches(self) -> List[Dict[str, Any]]:
        """List all batches."""
        vectorstore = self.get_vectorstore()

        try:
            # Get all batch documents
            results = vectorstore.similarity_search(
                "",
                k=1000,  # Large number to get all batches
                filter={"type": "batch"}
            )
            return [doc.metadata for doc in results]
        except:
            return []

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

        # Update batch artifact count
        if documents:
            batch_id = documents[0].metadata.get("batch_id")
            if batch_id:
                self._update_batch_artifact_count(batch_id)

        return ids

    def _update_batch_artifact_count(self, batch_id: int):
        """Update the artifact count for a batch."""
        # Get unique artifact IDs for this batch
        artifacts = self.list_artifacts_for_batch(batch_id)
        artifact_count = len(artifacts)

        # Update batch metadata (this is a simplified approach)
        # In a real implementation, you might want to update the batch document
        # For now, we'll just track this in the artifact metadata

    def get_artifact_by_id(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Get artifact information by artifact ID."""
        vectorstore = self.get_vectorstore()

        try:
            # Use ChromaDB where clause format for complex filters
            results = vectorstore.similarity_search(
                "document content text",  # Use a generic query
                k=100,
                filter={"$and": [{"type": {"$eq": "artifact_chunk"}}, {"artifact_id": {"$eq": artifact_id}}]}
            )
            if results:
                # Return the metadata from the first chunk (all chunks share core metadata)
                metadata = results[0].metadata.copy()

                # Count chunks for this artifact
                metadata["chunk_count"] = len(results)

                return metadata
        except Exception as e:
            print(f"Error getting artifact: {e}")
            # Fallback: try simpler filter
            try:
                results = vectorstore.similarity_search(
                    "document content text",
                    k=100,
                    filter={"artifact_id": artifact_id}
                )
                if results:
                    # Filter to only artifact chunks in code
                    artifact_chunks = [r for r in results if r.metadata.get("type") == "artifact_chunk"]
                    if artifact_chunks:
                        metadata = artifact_chunks[0].metadata.copy()
                        metadata["chunk_count"] = len(artifact_chunks)
                        return metadata
            except Exception as e2:
                print(f"Fallback error: {e2}")
        return None

    def list_artifacts(self) -> List[Dict[str, Any]]:
        """List all artifacts (grouped by artifact_id)."""
        vectorstore = self.get_vectorstore()

        try:
            # Get all artifact chunks
            results = vectorstore.similarity_search(
                "",
                k=10000,  # Large number to get all chunks
                filter={"type": "artifact_chunk"}
            )

            # Group by artifact_id
            artifacts = {}
            for doc in results:
                artifact_id = doc.metadata.get("artifact_id")
                if artifact_id and artifact_id not in artifacts:
                    metadata = doc.metadata.copy()
                    # Count chunks for this artifact
                    chunk_count = len([d for d in results if d.metadata.get("artifact_id") == artifact_id])
                    metadata["chunk_count"] = chunk_count
                    artifacts[artifact_id] = metadata

            return list(artifacts.values())
        except:
            return []

    def list_artifacts_for_batch(self, batch_id: int) -> List[Dict[str, Any]]:
        """List all artifacts for a specific batch."""
        vectorstore = self.get_vectorstore()

        try:
            results = vectorstore.similarity_search(
                "",
                k=10000,
                filter={"type": "artifact_chunk", "batch_id": batch_id}
            )

            # Group by artifact_id
            artifacts = {}
            for doc in results:
                artifact_id = doc.metadata.get("artifact_id")
                if artifact_id and artifact_id not in artifacts:
                    metadata = doc.metadata.copy()
                    chunk_count = len([d for d in results if d.metadata.get("artifact_id") == artifact_id])
                    metadata["chunk_count"] = chunk_count
                    artifacts[artifact_id] = metadata

            return list(artifacts.values())
        except:
            return []

    def search_similar(self, query: str, k: int = 5, filter_dict: Optional[dict] = None) -> List[Document]:
        """Search for similar documents in the vectorstore."""
        vectorstore = self.get_vectorstore()

        # Always filter out batch documents from similarity search
        base_filter = {"type": {"$eq": "artifact_chunk"}}

        if filter_dict:
            # Convert simple filter dict to ChromaDB format
            chroma_filters = []
            chroma_filters.append(base_filter)

            for key, value in filter_dict.items():
                chroma_filters.append({key: {"$eq": value}})

            search_filter = {"$and": chroma_filters}
        else:
            search_filter = base_filter

        return vectorstore.similarity_search(query, k=k, filter=search_filter)

    def search_by_batch(self, query: str, batch_id: int, k: int = 5) -> List[Document]:
        """Search for documents within a specific batch."""
        return self.search_similar(query, k=k, filter_dict={"batch_id": batch_id})

    def search_by_artifact(self, query: str, artifact_id: str, k: int = 5) -> List[Document]:
        """Search for documents within a specific artifact."""
        return self.search_similar(query, k=k, filter_dict={"artifact_id": artifact_id})

    def get_all_chunks_for_artifact(self, artifact_id: str) -> List[Document]:
        """Get all chunks for a specific artifact."""
        vectorstore = self.get_vectorstore()

        try:
            results = vectorstore.similarity_search(
                "document content text",  # Use generic query
                k=10000,
                filter={"$and": [{"type": {"$eq": "artifact_chunk"}}, {"artifact_id": {"$eq": artifact_id}}]}
            )
            # Sort by chunk_index
            results.sort(key=lambda x: x.metadata.get("chunk_index", 0))
            return results
        except Exception as e:
            print(f"Error getting chunks: {e}")
            # Fallback approach
            try:
                results = vectorstore.similarity_search(
                    "document content text",
                    k=10000,
                    filter={"artifact_id": artifact_id}
                )
                # Filter manually
                artifact_chunks = [r for r in results if r.metadata.get("type") == "artifact_chunk"]
                artifact_chunks.sort(key=lambda x: x.metadata.get("chunk_index", 0))
                return artifact_chunks
            except:
                return []

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
