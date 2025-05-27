import os
import re
import traceback
import asyncio
import json
import uuid
from langchain_aws import ChatBedrock
from langchain.chains import RetrievalQA
from src.config import settings
from src.utils.document_loader import load_projects
from src.utils.vector_store import VectorStore
from src.rag.prompt import get_qa_prompt
from src.utils.logger import setup_logger
from src.utils.connect_llm import get_llm

class RoleRAG:
    """
    A class to implement a Retrieval Augmented Generation (RAG) system 
    for extracting project details and activities performed for a specific role.
    Processes a specific JSON file and maintains a unique vector store index.
    """
    def __init__(self, index_id: str = 'role_rag'):
        self.config = settings
        self.logger = setup_logger()
        self.documents = []
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.json_file = self.config.projects_dir+"/role-rag.json"
        self.index_id = index_id 
        self.index_path = os.path.join(self.config.faiss_index_dir, f"faiss_index_{self.index_id}")
        
        # Create necessary directories
        os.makedirs(self.config.projects_dir, exist_ok=True)
        os.makedirs(self.config.faiss_index_dir, exist_ok=True)

    def _initialize_llm(self):
        """Initializes the AWS Bedrock LLM (ChatBedrock)."""
        try:
            self.logger.info(f"Initializing ChatBedrock LLM: {self.config.llm_model_name}")
            self.llm = get_llm()
            self.logger.info("ChatBedrock LLM initialized successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing ChatBedrock LLM: {e}")
            traceback.print_exc()
            return False

    async def load_projects(self) -> bool:
        """Loads project data from the specified JSON file."""
        try:
            self.documents = await asyncio.to_thread(
                load_projects,
                self.config.projects_dir,
                self.json_file
            )
            if not self.documents:
                self.logger.error(f"No documents loaded for JSON file: {self.json_file}.")
                return False
            self.logger.info(f"Loaded {len(self.documents)} project documents for {self.json_file}.")
            return True
        except Exception as e:
            self.logger.error(f"Error loading projects from '{self.json_file}': {e}")
            return False

    async def setup_vectorstore(self, force_rebuild: bool = False) -> bool:
        """Sets up the FAISS vector store using the VectorStore class."""
        if not self.documents:
            self.logger.error("No documents loaded. Cannot setup vector store.")
            return False

        try:
            self.vector_store = VectorStore(
                model_name=self.config.embedding_model_name,
                model_cache_dir=self.config.model_cache_dir,
                index_path=self.index_path,
                index_id=self.index_id
            )
            success = await self.vector_store.load_or_create(self.documents, force_rebuild)
            if not success:
                self.logger.error("Failed to setup vector store.")
                return False
            self.logger.info(f"Vector store setup complete for index: {self.index_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error setting up vector store: {e}")
            traceback.print_exc()
            return False

    async def setup_qa_chain(self):
        """Sets up the RetrievalQA chain with a prompt for structured output."""
        if not self.vector_store or not self.vector_store.get_vectorstore():
            self.logger.error("Vector store not available. Cannot setup QA chain.")
            return False
        if not self.llm:
            self.logger.error("LLM not initialized. Cannot setup QA chain.")
            return False

        try:
            QA_PROMPT = get_qa_prompt()
            self.qa_chain = await asyncio.to_thread(
                RetrievalQA.from_chain_type,
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.get_vectorstore().as_retriever(
                    search_kwargs={"k": self.config.retriever_k}
                ),
                chain_type_kwargs={"prompt": QA_PROMPT},
                return_source_documents=True
            )
            self.logger.info("QA chain created with structured output prompt.")
            return True
        except Exception as e:
            self.logger.error(f"Error creating QA chain: {e}")
            traceback.print_exc()
            return False

    async def query(self,position: str) -> dict | None:
        """Performs a query against the RAG system using project_name and position."""
        question = f"What activities does the {position} perform for different project?"
        if not self.qa_chain:
            self.logger.error("QA chain is not initialized. Call setup_qa_chain() first.")
            return None
        
        try:
            result = await asyncio.to_thread(
                self.qa_chain.invoke,
                {"query": question}
            )
            response_text = result['result'].strip()
            
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1).strip()
            
            try:
                structured_response = json.loads(response_text)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error: LLM response is not valid JSON. Raw response: {result['result']}")
                structured_response = {
                    "Position held": position,
                    "Activities performed": "Insufficient details provided to determine specific activities for this role."
                }
            
            return {
                "response": structured_response,
                "source_documents": result.get('source_documents', [])
            }
        except Exception as e:
            self.logger.error(f"Error during QA chain invocation for query '{question}': {e}")
            traceback.print_exc()
            return None

    async def full_setup(self, force_rebuild_index: bool = False) -> bool:
        """Performs all setup steps: load projects, setup vector store, and setup QA chain."""
        try:
            self.logger.info(f"Starting full RAG setup for JSON file: {self.json_file}...")
            
            if not self.llm:
                self.logger.info("Initializing LLM...")
                if not self._initialize_llm():
                    self.logger.error("Failed to initialize LLM")
                    return False
            
            if not self.documents:
                self.logger.info(f"Loading projects from '{self.json_file}'...")
                if not await self.load_projects():
                    self.logger.error("Failed to load projects")
                    return False
            
            if not self.vector_store or not self.vector_store.get_vectorstore():
                self.logger.info("Setting up vectorstore...")
                if not await self.setup_vectorstore(force_rebuild=force_rebuild_index):
                    self.logger.error("Failed to setup vectorstore")
                    return False
            
            if not self.qa_chain:
                self.logger.info("Setting up QA chain...")
                if not await self.setup_qa_chain():
                    self.logger.error("Failed to setup QA chain")
                    return False
            
            self.logger.info(f"ProjectRAG system fully set up for JSON file: {self.json_file}.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during full setup: {e}")
            traceback.print_exc()
            return False

    def get_model_info(self) -> dict:
        """Get information about the loaded models and system status."""
        return {
            "status": "initialized" if self.llm and self.vector_store and self.vector_store.get_vectorstore() else "partially initialized",
            "json_file": self.json_file,
            "index_id": self.index_id,
            "embedding_model": {
                "loaded": self.vector_store and self.vector_store.get_embedding_model() is not None,
                "name": self.config.embedding_model_name,
                "cache_dir": self.config.model_cache_dir
            },
            "llm": {
                "loaded": self.llm is not None,
                "name": self.config.llm_model_name
            },
            "vectorstore": {
                "loaded": self.vector_store and self.vector_store.get_vectorstore() is not None,
                "documents_count": len(self.documents) if self.documents else 0,
                "index_path": self.index_path
            },
            "qa_chain": {
                "ready": self.qa_chain is not None
            }
        }

    async def cleanup(self):
        """Cleanup resources."""
        try:
            self.logger.info(f"Cleaning up ProjectRAG resources for JSON file: {self.json_file}...")
            self.qa_chain = None
            self.vector_store = None
            self.documents = []
            self.logger.info("ProjectRAG cleanup completed.")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            traceback.print_exc()

    def is_ready(self) -> bool:
        """Check if the RAG system is ready to handle queries."""
        return all([
            self.llm is not None,
            self.vector_store is not None,
            self.vector_store.get_vectorstore() is not None,
            self.qa_chain is not None,
            len(self.documents) > 0
        ])