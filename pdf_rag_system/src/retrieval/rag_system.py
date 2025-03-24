import os
from typing import List, Optional
import shutil

from llama_index.core import (
    VectorStoreIndex, 
    StorageContext,
    Settings,
    Document,
    load_index_from_storage
)
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

from src.processor.pdf_processor import PDFProcessor
from src.llm.openai_integration import OPENAI_AVAILABLE
from src.config import config

class RAGSystem:
    def __init__(self, pdf_dir: str = None, persist_directory: str = None):
        # Use config values with fallback to parameters
        self.pdf_dir = pdf_dir or config.DATA_DIR
        self.persist_directory = persist_directory or config.VECTOR_STORE_DIR
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.pdf_processor = PDFProcessor(self.pdf_dir)
        
        # Set up LlamaIndex Settings
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=config.EMBEDDING_MODEL
        )
        
        # Initialize the LLM if OpenAI is available
        if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            try:
                Settings.llm = OpenAI(temperature=config.LLM_TEMPERATURE)
                print("Successfully initialized OpenAI LLM")
            except Exception as e:
                print(f"Error initializing OpenAI: {e}")
                Settings.llm = None
                print("Falling back to basic document retrieval")
        else:
            Settings.llm = None
        
        self.index = None
        self.chat_engine = None
        self.chat_history = []
    
    def _load_existing_index(self) -> bool:
        """Try to load an existing index from storage"""
        if os.path.exists(self.persist_directory):
            try:
                print(f"Loading existing index from {self.persist_directory}...")
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_directory)
                self.index = load_index_from_storage(storage_context)
                print("Successfully loaded existing index!")
                return True
            except Exception as e:
                print(f"Error loading existing index: {e}")
        return False
    
    def initialize(self, force_reload: bool = False) -> bool:
        """Initialize the RAG system by loading and indexing PDFs"""
        # If not force_reload, try to load existing index first
        if not force_reload and self._load_existing_index():
            # Initialize the chat engine with the loaded index
            self._initialize_chat_engine()
            return True
            
        # If force_reload, remove the existing vector store directory
        if force_reload and os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                print(f"Removed existing vector store at {self.persist_directory}")
                os.makedirs(self.persist_directory, exist_ok=True)
            except Exception as e:
                print(f"Error removing existing vector store: {e}")
        
        # Check if there are PDFs to process
        try:
            if force_reload:
                pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
            else:
                pdf_files = self.pdf_processor.get_unprocessed_pdfs()
                
            if not pdf_files and not force_reload:
                print("No new PDF files to process.")
                print("Checking for existing index...")
                if self._load_existing_index():
                    self._initialize_chat_engine()
                    return True
                else:
                    print("No existing index found. Please add PDFs to process.")
                    return False
                
        except FileNotFoundError as e:
            print(str(e))
            return False
        
        # Load PDFs if needed
        documents = self.pdf_processor.load_pdfs(force_reload=force_reload)
        
        if not documents:
            print("No documents loaded. Please add PDFs to the data directory.")
            return False
        
        # Split the documents into chunks
        nodes = self.pdf_processor.split_documents(documents)
        
        if not nodes:
            print("No nodes created.")
            return False
        
        # Create the vector store from scratch
        print("Creating vector store...")
        try:
            # Create a simple index with just the documents
            self.index = VectorStoreIndex.from_documents(nodes)
            
            # Save the index to disk
            self.index.storage_context.persist(persist_dir=self.persist_directory)
            print("Vector store created and persisted successfully!")
            
            # Initialize the chat engine
            self._initialize_chat_engine()
            return True
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            print("Trying alternative approach...")
            
            try:
                # Try a different approach if the first one fails
                storage_context = StorageContext.from_defaults()
                self.index = VectorStoreIndex([], storage_context=storage_context)
                
                # Add documents one by one
                for doc in documents:
                    try:
                        self.index.insert(doc)
                    except Exception as doc_error:
                        print(f"Error adding document: {doc_error}")
                
                # Save the index to disk
                storage_context.persist(persist_dir=self.persist_directory)
                print("Vector store created using alternative approach!")
                
                # Initialize the chat engine
                self._initialize_chat_engine()
                return True
                
            except Exception as alt_error:
                print(f"Alternative approach also failed: {alt_error}")
                return False
    
    def _initialize_chat_engine(self):
        """Initialize the chat engine for conversational retrieval"""
        if self.index:
            try:
                # Create a chat engine with conversation history
                self.chat_engine = self.index.as_chat_engine(
                    verbose=True,
                    system_prompt=(
                        "You are a helpful assistant that answers questions based on the provided documents. "
                        "If you don't know the answer, just say that you don't know."
                    )
                )
                print("Chat engine initialized successfully")
            except Exception as e:
                print(f"Error initializing chat engine: {e}")
                print("Will use basic query engine instead")
    
    def query(self, question: str, k: int = None) -> str:
        """Query the RAG system with a question"""
        # Use the config value if k is not provided
        k = k or config.DEFAULT_RETRIEVAL_K
        
        if not self.index:
            return "Error: Vector store not initialized. Please run initialize() first."
        
        # Create a user message with the question
        user_message = ChatMessage(role="user", content=question)
        
        # Add message to history if it's enabled
        if config.ENABLE_CHAT_HISTORY:
            self.chat_history.append(user_message)
        
        try:
            # If we have a chat engine, use it for a conversational response
            if self.chat_engine and Settings.llm:
                try:
                    response = self.chat_engine.chat(message=question)
                    response_text = response.response
                except Exception as chat_error:
                    print(f"Error using chat engine: {chat_error}")
                    print("Falling back to basic query...")
                    query_engine = self.index.as_query_engine()
                    response = query_engine.query(question)
                    response_text = str(response)
                
                # Add the response to chat history if enabled
                if config.ENABLE_CHAT_HISTORY:
                    self.chat_history.append(ChatMessage(role="assistant", content=response_text))
                    
                    # Trim history if it exceeds the max messages
                    if len(self.chat_history) > config.CHAT_HISTORY_MAX_MESSAGES * 2:
                        self.chat_history = self.chat_history[-config.CHAT_HISTORY_MAX_MESSAGES * 2:]
                
                return response_text
            
            # If no chat engine or LLM, use basic retrieval
            return self._fallback_answer(question, k)
        
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return f"An error occurred: {str(e)}"
    
    def _fallback_answer(self, question: str, k: int) -> str:
        """Fallback method to return relevant excerpts when LLM is not available"""
        try:
            # Try to use the retriever
            query_engine = self.index.as_query_engine()
            response = query_engine.query(question)
            return str(response)
        except Exception as e:
            print(f"Error using query engine: {e}")
            return f"Could not retrieve relevant information: {str(e)}"
    
    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history = []
        print("Chat history cleared.")