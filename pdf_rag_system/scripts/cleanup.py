#!/usr/bin/env python3
import os
import shutil
import argparse
import json
from typing import List, Optional

def delete_vector_database(db_path: str = "storage/chroma_db") -> bool:
    """Delete the Chroma vector database"""
    if os.path.exists(db_path):
        try:
            print(f"Deleting vector database at {db_path}...")
            shutil.rmtree(db_path)
            print(f"Vector database deleted successfully.")
            return True
        except Exception as e:
            print(f"Error deleting vector database: {e}")
            return False
    else:
        print(f"Vector database not found at {db_path}")
        return False

def delete_processed_pdfs_record(record_file: str = "storage/processed_pdfs.json") -> bool:
    """Delete the processed PDFs record file"""
    if os.path.exists(record_file):
        try:
            print(f"Deleting processed PDFs record at {record_file}...")
            os.remove(record_file)
            print(f"Processed PDFs record deleted successfully.")
            return True
        except Exception as e:
            print(f"Error deleting processed PDFs record: {e}")
            return False
    else:
        print(f"Processed PDFs record not found at {record_file}")
        return False

def delete_pdf_files(pdf_dir: str = "data", confirm: bool = True) -> bool:
    """Delete all PDF files in the specified directory"""
    if not os.path.exists(pdf_dir):
        print(f"PDF directory not found at {pdf_dir}")
        return False
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return False
    
    if confirm:
        confirmation = input(f"Are you sure you want to delete {len(pdf_files)} PDF files? (y/n): ")
        if confirmation.lower() != 'y':
            print("PDF deletion cancelled.")
            return False
    
    success = True
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        try:
            os.remove(pdf_path)
            print(f"Deleted {pdf_file}")
        except Exception as e:
            print(f"Error deleting {pdf_file}: {e}")
            success = False
    
    if success:
        print(f"All PDF files deleted successfully.")
    else:
        print(f"Some PDF files could not be deleted.")
    
    return success

def list_resources() -> None:
    """List resources that can be cleaned up"""
    resources = []
    
    # Check for vector database
    if os.path.exists("storage/chroma_db"):
        size = get_directory_size("storage/chroma_db")
        resources.append(("Vector database", "storage/chroma_db", f"{size:.2f} MB"))
    
    # Check for processed PDFs record
    if os.path.exists("storage/processed_pdfs.json"):
        size_kb = os.path.getsize("storage/processed_pdfs.json") / 1024
        resources.append(("Processed PDFs record", "storage/processed_pdfs.json", f"{size_kb:.2f} KB"))
    
    # Check for PDF files
    if os.path.exists("data"):
        pdf_files = [f for f in os.listdir("data") if f.lower().endswith('.pdf')]
        if pdf_files:
            total_size = sum(os.path.getsize(os.path.join("data", f)) for f in pdf_files) / (1024 * 1024)
            resources.append((f"PDF files ({len(pdf_files)})", "data/*.pdf", f"{total_size:.2f} MB"))
    
    if resources:
        print("Available resources to clean up:")
        for i, (name, path, size) in enumerate(resources, 1):
            print(f"{i}. {name} at {path} ({size})")
    else:
        print("No resources found to clean up.")

def get_directory_size(path: str) -> float:
    """Get the size of a directory in megabytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert to MB

def main():
    parser = argparse.ArgumentParser(description="Clean up RAG system resources")
    parser.add_argument("--list", action="store_true", help="List resources that can be cleaned up")
    parser.add_argument("--delete-all", action="store_true", help="Delete all resources (vector DB, records, and PDFs)")
    parser.add_argument("--delete-db", action="store_true", help="Delete the vector database")
    parser.add_argument("--delete-records", action="store_true", help="Delete the processed PDFs record")
    parser.add_argument("--delete-pdfs", action="store_true", help="Delete all PDF files")
    parser.add_argument("--db-path", default="storage/chroma_db", help="Path to the vector database")
    parser.add_argument("--record-file", default="storage/processed_pdfs.json", help="Path to the processed PDFs record file")
    parser.add_argument("--pdf-dir", default="data", help="Directory containing PDF files")
    parser.add_argument("--force", action="store_true", help="Force deletion without confirmation")
    
    args = parser.parse_args()
    
    if args.list:
        list_resources()
        return
    
    if not any([args.delete_all, args.delete_db, args.delete_records, args.delete_pdfs]):
        parser.print_help()
        return
    
    if args.delete_all:
        args.delete_db = args.delete_records = args.delete_pdfs = True
    
    if args.delete_db:
        delete_vector_database(args.db_path)
    
    if args.delete_records:
        delete_processed_pdfs_record(args.record_file)
    
    if args.delete_pdfs:
        delete_pdf_files(args.pdf_dir, not args.force)

if __name__ == "__main__":
    main()