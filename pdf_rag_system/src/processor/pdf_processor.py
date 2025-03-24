import os
import hashlib
import json
import time
from typing import List, Dict, Optional

from pypdf import PdfReader
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from src.config import config

class PDFProcessor:
    def __init__(self, pdf_dir: str = None, processed_pdfs_file: str = None):
        # Use config values with fallback to parameters
        self.pdf_dir = pdf_dir or config.DATA_DIR
        self.processed_pdfs_file = processed_pdfs_file or config.PROCESSED_PDFS_FILE
        self.documents = []
        
        # Create storage directory if it doesn't exist
        os.makedirs(os.path.dirname(self.processed_pdfs_file), exist_ok=True)
        
        self.processed_pdfs = self._load_processed_pdfs()
        
        self.text_splitter = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
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
        
        all_documents = []
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
                        # Create LlamaIndex Document
                        doc = Document(
                            text=text,
                            metadata={
                                "source": pdf_file, 
                                "page": i,
                                "version_id": file_hash,
                                "timestamp": time.time(),
                                "status": "current"
                            }
                        )
                        all_documents.append(doc)
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
        
        # Update processed PDFs record
        self.processed_pdfs.update(newly_processed)
        self._save_processed_pdfs()
        
        if all_documents:
            print(f"Extracted {len(all_documents)} pages from {len(pdf_files)} PDF files")
        
        self.documents.extend(all_documents)
        return all_documents
    
    def split_documents(self, documents: Optional[List[Document]] = None) -> List[Document]:
        """Split documents into chunks"""
        if documents is None:
            documents = self.documents
            
        if not documents:
            print("No documents to split.")
            return []
        
        # Use the correct method depending on your LlamaIndex version
        try:
            # Method for newer versions
            nodes = self.text_splitter.get_nodes_from_documents(documents)
        except AttributeError:
            try:
                # Method for older versions
                nodes = self.text_splitter.split_documents(documents)
            except Exception as e:
                print(f"Error splitting documents: {e}")
                return documents  # Return original documents if splitting fails
        
        print(f"Split into {len(nodes)} chunks")
        return nodes