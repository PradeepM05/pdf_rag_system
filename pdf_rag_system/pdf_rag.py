import os
import sys
import hashlib
import json
import time
from typing import List, Dict, Set, Optional
import argparse
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Load environment variables from .env file if it exists
load_dotenv()

# Print debugging information about the OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY", "")
if api_key:
    print(f"OpenAI API key found: {api_key[:5]}...{api_key[-4:]}")
else:
    print("WARNING: No OpenAI API key found in environment variables")

# Try to import OpenAI for completions with more compatible imports
try:
    from langchain_openai import ChatOpenAI  # Use ChatOpenAI instead of OpenAI
    from langchain.chains import RetrievalQA  # Use RetrievalQA as a fallback
    from langchain_core.prompts import PromptTemplate
    OPENAI_AVAILABLE = True
    print("Successfully imported OpenAI and related packages")
except ImportError as e:
    OPENAI_AVAILABLE = False
    print(f"OpenAI package not available: {e}")
    print("Using fallback for completions.")

class PDFProcessor:
    def __init__(self, pdf_dir: str, processed_pdfs_file: str = "processed_pdfs.json"):
        self.pdf_dir = pdf_dir
        self.documents = []
        self.processed_pdfs_file = processed_pdfs_file
        self.processed_pdfs = self._load_processed_pdfs()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute a hash of the file to detect changes"""
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    
    def _load_processed_pdfs(self) -> Dict[str, str]:
        """Load the list of already processed PDFs and their hashes"""
        if os.path.exists(self.processed_pdfs_file):
            try:
                with open(self.processed_pdfs_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not load processed PDFs file. Starting fresh.")
        return {}
    
    def _save_processed_pdfs(self):
        """Save the list of processed PDFs to a file"""
        with open(self.processed_pdfs_file, 'w') as f:
            json.dump(self.processed_pdfs, f)
    
    def get_unprocessed_pdfs(self) -> List[str]:
        """Get list of PDFs that need processing (new or modified)"""
        if not os.path.exists(self.pdf_dir):
            raise FileNotFoundError(f"Directory not found: {self.pdf_dir}")
        
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_dir}")
            return []
        
        unprocessed_pdfs = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            file_hash = self._compute_file_hash(pdf_path)
            
            # Check if the PDF was processed before and has the same hash
            if pdf_file not in self.processed_pdfs or self.processed_pdfs[pdf_file] != file_hash:
                unprocessed_pdfs.append(pdf_file)
        
        return unprocessed_pdfs
    
    def load_pdfs(self, force_reload: bool = False) -> List[Document]:
        """Load all PDFs from the directory or only new/modified ones"""
        if not os.path.exists(self.pdf_dir):
            raise FileNotFoundError(f"Directory not found: {self.pdf_dir}")
        
        # Get list of PDFs to process
        if force_reload:
            pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
        else:
            pdf_files = self.get_unprocessed_pdfs()
        
        if not pdf_files:
            print(f"No new or modified PDF files to process in {self.pdf_dir}")
            return []
        
        all_texts = []
        newly_processed = {}
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            print(f"Processing {pdf_file}...")
            
            try:
                # Compute hash for tracking
                file_hash = self._compute_file_hash(pdf_path)
                newly_processed[pdf_file] = file_hash
                
                # Extract text from PDF
                reader = PdfReader(pdf_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty text
                        all_texts.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": pdf_file, 
                                    "page": i,
                                    # Add version metadata
                                    "version_id": file_hash,
                                    "timestamp": time.time(),
                                    "status": "current"
                                }
                            )
                        )
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
        
        # Update processed PDFs record
        self.processed_pdfs.update(newly_processed)
        self._save_processed_pdfs()
        
        if all_texts:
            print(f"Extracted {len(all_texts)} pages from {len(pdf_files)} PDF files")
        
        self.documents.extend(all_texts)
        return all_texts
    
    def split_documents(self, documents: List[Document] = None) -> List[Document]:
        """Split documents into chunks"""
        if documents is None:
            documents = self.documents
            
        if not documents:
            print("No documents to split.")
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks

class RAGSystem:
    def __init__(self, pdf_dir: str, persist_directory: str = "chroma_db"):
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
                self.llm = ChatOpenAI(temperature=0)
                # Create a simple prompt template
                prompt_template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""
                # We'll use RetrievalQA as it's more compatible across versions
                self.qa_chain = None  # We'll create this during querying
                print("Successfully initialized OpenAI LLM")
            except Exception as e:
                print(f"Error initializing OpenAI: {e}")
                self.llm = None
                self.qa_chain = None
                print("Falling back to basic document retrieval")
        else:
            self.llm = None
            self.qa_chain = None
    
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
                old_docs = self.vector_store.get(
                    where={"source": pdf_file, "status": "current"}
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
    
    def query(self, question: str, k: int = 4) -> str:
        """Query the RAG system with a question"""
        if not self.vector_store:
            return "Error: Vector store not initialized. Please run initialize() first."
        
        # Retrieve relevant documents - only get current status documents
        try:
            docs = self.vector_store.similarity_search(
                question, 
                k=k,
                filter={"status": "current"}  # Only get current documents
            )
            
            if not docs:
                return "No relevant information found."
            
            # If we have an LLM, try to use it
            if self.llm and OPENAI_AVAILABLE:
                try:
                    # Create a retriever
                    retriever = self.vector_store.as_retriever(
                        search_kwargs={"k": k, "filter": {"status": "current"}}
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

def main():
    parser = argparse.ArgumentParser(description="Simple PDF RAG System")
    parser.add_argument("--pdf_dir", default="data", help="Directory containing PDF files")
    parser.add_argument("--force_reload", action="store_true", help="Force reload of all PDFs, ignoring the processed list")
    parser.add_argument("--test_openai", action="store_true", help="Test OpenAI connectivity")
    args = parser.parse_args()
    
    # Test OpenAI connectivity if requested
    if args.test_openai:
        if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI
                test_llm = ChatOpenAI(temperature=0)
                response = test_llm.invoke("This is a test message. Reply with 'OpenAI connection successful.'")
                print(f"OpenAI Test Result: {response.content}")
                print("OpenAI connectivity test completed successfully.")
            except Exception as e:
                print(f"OpenAI test failed: {e}")
        else:
            print("OpenAI packages not available or API key not set. Cannot test.")
    
    rag_system = RAGSystem(args.pdf_dir)
    
    if not rag_system.initialize(force_reload=args.force_reload):
        print("Failed to initialize RAG system.")
        return
    
    print("\nRAG system initialized successfully!")
    print("You can now ask questions about the PDFs.")
    print("Type 'exit' to quit.")
    
    while True:
        question = input("\nQuestion: ")
        
        if question.lower() in ('exit', 'quit'):
            break
            
        answer = rag_system.query(question)
        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    main()