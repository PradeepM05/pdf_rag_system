import os
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from src.processor.pdf_processor import PDFProcessor
from src.llm.openai_integration import OPENAI_AVAILABLE
from src.config import config  # Import the centralized configuration


class RAGSystem:
    def __init__(self, pdf_dir: str, persist_directory: str = "storage/chroma_db"):
        # Create storage directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        self.pdf_processor = PDFProcessor(pdf_dir)
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        
        # Initialize the LLM if OpenAI is available
        if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            try:
                # Use ChatOpenAI instead of deprecated OpenAI
                self.llm = ChatOpenAI(temperature=config.LLM_TEMPERATURE)
                print("Successfully initialized OpenAI LLM")
            except Exception as e:
                print(f"Error initializing OpenAI: {e}")
                self.llm = None
                print("Falling back to basic document retrieval")
        else:
            self.llm = None
    
    def _load_existing_vectorstore(self) -> bool:
        """Attempt to load an existing Chroma DB"""
        if os.path.exists(self.persist_directory):
            try:
                print("Loading existing vector database...")
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                return True
            except Exception as e:
                print(f"Error loading existing vector database: {e}")
        return False
    
    def initialize(self, force_reload: bool = False) -> bool:
        """Initialize the RAG system by loading and indexing PDFs"""
        # First try to load the existing vector store
        vector_store_exists = self._load_existing_vectorstore()
        
        # Check if there are new PDFs to process
        try:
            unprocessed_pdfs = self.pdf_processor.get_unprocessed_pdfs()
        except FileNotFoundError as e:
            print(str(e))
            return False
        
        # If we have a vector store and no new PDFs, we're done
        if vector_store_exists and not unprocessed_pdfs and not force_reload:
            print("Using existing vector database. No new documents to process.")
            return True
        
        # Load new PDFs if any
        documents = self.pdf_processor.load_pdfs(force_reload=force_reload)
        
        # If we have new PDFs to process and vector store exists
        if documents and vector_store_exists:
            for pdf_file in unprocessed_pdfs:
                print(f"Marking old chunks as deprecated for: {pdf_file}")
                # Get the new file hash
                new_hash = self.pdf_processor.processed_pdfs.get(pdf_file)
                
                # First mark old chunks as deprecated (instead of deleting)
                # Use the correct filter syntax for ChromaDB
                try:
                    # Use the $and operator for multiple conditions
                    where_filter = {
                        "$and": [
                            {"source": {"$eq": pdf_file}},
                            {"status": {"$eq": "current"}}
                        ]
                    }
                    
                    old_docs = self.vector_store.get(
                        where=where_filter
                    )
                    
                    if old_docs and hasattr(self.vector_store, '_collection'):
                        # Get IDs of old documents to update
                        old_ids = [doc.id for doc in old_docs]
                        
                        # Update their status to "deprecated"
                        self.vector_store._collection.update(
                            ids=old_ids,
                            metadatas=[
                                {"source": pdf_file, "page": doc.metadata["page"], 
                                "version_id": doc.metadata["version_id"],
                                "timestamp": doc.metadata["timestamp"], 
                                "status": "deprecated"}
                                for doc in old_docs
                            ]
                        )
                        print(f"Marked {len(old_ids)} chunks as deprecated")
                except Exception as e:
                    print(f"Error marking old chunks as deprecated: {e}")
                    print("Continuing with new document processing...")
        
        # If no documents loaded or found, but we have an existing vector store, still return success
        if not documents and vector_store_exists:
            print("No new documents to process. Using existing vector store.")
            return True
        
        # If no documents and no existing vector store, return failure
        if not documents and not vector_store_exists:
            print("No documents loaded. Please add PDFs to the data directory.")
            return False
        
        # Split the documents into chunks
        chunks = self.pdf_processor.split_documents(documents)
        
        if not chunks:
            print("No chunks created.")
            return False
        
        # Create or update the vector store
        print("Creating/updating vector store...")
        if vector_store_exists:
            # Add new documents to existing DB
            self.vector_store.add_documents(chunks)
            print("Vector store updated successfully!")
        else:
            # Create new DB
            self.vector_store = Chroma.from_documents(
                documents=chunks, 
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print("Vector store created successfully!")
        
        # Persist the changes
        if hasattr(self.vector_store, '_persist'):
            self.vector_store.persist()
            print("Vector store persisted to disk")
        
        return True
    
    def query(self, question: str, k: int = None) -> str:
        """Query the RAG system with a question"""
        # Use the config value if k is not provided
        k = k or config.DEFAULT_RETRIEVAL_K
        if not self.vector_store:
            return "Error: Vector store not initialized. Please run initialize() first."
        
        # Retrieve relevant documents - only get current status documents
        try:
            # Use the correct filter syntax for ChromaDB
            where_filter = {"status": {"$eq": "current"}}
            
            docs = self.vector_store.similarity_search(
                question, 
                k=k or config.DEFAULT_RETRIEVAL_K,
                filter=where_filter
            )
            
            if not docs:
                return "No relevant information found."
            
            # If we have an LLM, try to use it
            if self.llm and OPENAI_AVAILABLE:
                try:
                    # Create a retriever with the correct filter syntax
                    retriever = self.vector_store.as_retriever(
                        search_kwargs={"k": k, "filter": where_filter}
                    )
                    
                    # Create a RetrievalQA chain
                    qa = RetrievalQA.from_chain_type(
                        llm=self.llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=False
                    )
                    
                    # Execute the chain
                    result = qa.invoke({"query": question})
                    
                    # Check if result is a string or a dict (different versions return different types)
                    if isinstance(result, dict) and "result" in result:
                        return result["result"]
                    elif isinstance(result, str):
                        return result
                    else:
                        return str(result)
                    
                except Exception as e:
                    print(f"Error using QA chain: {e}")
                    print("Falling back to basic retrieval...")
                    return self._fallback_answer(docs, question)
            
            # Fallback if no LLM is available
            return self._fallback_answer(docs, question)
        
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return f"An error occurred: {str(e)}"
    
    def _fallback_answer(self, docs: List[Document], question: str) -> str:
        """Fallback method to return relevant excerpts when LLM is not available"""
        if not docs:
            return "No relevant information found."
        
        result = "Here are some relevant excerpts from the documents:\n\n"
        for i, doc in enumerate(docs, 1):
            result += f"Excerpt {i} (from {doc.metadata['source']}, page {doc.metadata['page']}):\n"
            result += f"{doc.page_content}\n\n"
        
        return result