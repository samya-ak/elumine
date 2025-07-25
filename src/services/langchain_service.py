"""LangChain service for text processing and embeddings."""

import os
from typing import List, Optional
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from src.config import config_manager
from src.db.chroma import get_chroma_path

class LangChainService:
    """Service for LangChain operations including chunking and embeddings."""

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
                collection_name="elumine_artifacts"
            )
        return self._vectorstore

    def create_documents(
        self,
        text: str,
        metadata: dict,
        filename: str,
        batch_id: int
    ) -> List[Document]:
        """Create LangChain documents from text with metadata."""
        # Create initial document with base metadata
        base_metadata = {
            **metadata,
            "filename": filename,
            "batch_id": batch_id,
        }

        initial_doc = Document(page_content=text, metadata=base_metadata)

        # Use split_documents - preserves metadata automatically
        documents = self.text_splitter.split_documents([initial_doc])

        # Add chunk-specific metadata
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "chunk_index": i,
                "total_chunks": len(documents),
                "chunk_id": f"artifact_{batch_id}_{Path(filename).stem}_chunk_{i}"
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
_langchain_service = None

def get_langchain_service() -> LangChainService:
    """Get the singleton LangChain service instance."""
    global _langchain_service
    if _langchain_service is None:
        _langchain_service = LangChainService()
    return _langchain_service
