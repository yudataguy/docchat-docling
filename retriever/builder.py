from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config.settings import settings
import hashlib
import os
import logging

logger = logging.getLogger(__name__)

# Batch size for embedding (stay well under 300k token limit)
EMBEDDING_BATCH_SIZE = 500
VECTOR_STORE_DIR = "vector_cache"


class RetrieverBuilder:
    def __init__(self):
        """Initialize the retriever builder with OpenAI embeddings."""
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.OPENAI_API_KEY
        )
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    def _get_docs_hash(self, docs) -> str:
        """Generate hash from document contents for cache key."""
        content = "".join(doc.page_content for doc in docs)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def build_hybrid_retriever(self, docs):
        """Build a hybrid retriever using BM25 and vector-based retrieval."""
        try:
            docs_hash = self._get_docs_hash(docs)
            persist_path = os.path.join(VECTOR_STORE_DIR, docs_hash)

            # Check if cached vector store exists
            if os.path.exists(persist_path):
                logger.info(f"Loading cached vector store from {persist_path}")
                vector_store = Chroma(
                    persist_directory=persist_path,
                    embedding_function=self.embeddings
                )
            else:
                # Create new vector store in batches
                logger.info(f"Embedding {len(docs)} documents in batches of {EMBEDDING_BATCH_SIZE}...")

                vector_store = Chroma.from_documents(
                    documents=docs[:EMBEDDING_BATCH_SIZE],
                    embedding=self.embeddings,
                    persist_directory=persist_path
                )

                for i in range(EMBEDDING_BATCH_SIZE, len(docs), EMBEDDING_BATCH_SIZE):
                    batch = docs[i:i + EMBEDDING_BATCH_SIZE]
                    vector_store.add_documents(batch)
                    logger.info(f"Embedded batch {i // EMBEDDING_BATCH_SIZE + 1}/{(len(docs) - 1) // EMBEDDING_BATCH_SIZE + 1}")

            logger.info("Vector store ready.")
            
            # Create BM25 retriever
            bm25 = BM25Retriever.from_documents(docs)
            logger.info("BM25 retriever created successfully.")
            
            # Create vector-based retriever
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": settings.VECTOR_SEARCH_K})
            logger.info("Vector retriever created successfully.")
            
            # Combine retrievers into a hybrid retriever
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=settings.HYBRID_RETRIEVER_WEIGHTS
            )
            logger.info("Hybrid retriever created successfully.")
            return hybrid_retriever
        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise