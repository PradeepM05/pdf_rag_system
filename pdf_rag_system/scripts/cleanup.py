#!/usr/bin/env python3
import os
import shutil
import argparse
import json
import time
from typing import List, Optional
from pathlib import Path

def delete_vector_database(db_path: str = "storage/index_db") -> bool:
    """Delete the FAISS vector database"""
    if os.path.exists(db_path):
        try:
            print(f"Deleting FAISS vector database at {db_path}...")
            shutil.rmtree(db_path)
            print(f"FAISS vector database deleted successfully.")
            return True
        except Exception as e:
            print(f"Error deleting FAISS vector database: {e}")
            return False
    else:
        print(f"FAISS vector database not found at {db_path}")
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

def backup_vector_database(src_path: str, backup_dir: str) -> str:
    """Backup the vector database directory"""
    if not os.path.exists(src_path):
        print(f"Vector database not found at {src_path}")
        return ""
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create timestamped backup directory
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    backup_path = os.path.join(backup_dir, f"vectordb_backup_{timestamp}")
    
    try:
        print(f"Backing up vector database to {backup_path}...")
        shutil.copytree(src_path, backup_path)
        print(f"Backup completed successfully.")
        return backup_path
    except Exception as e:
        print(f"Backup failed: {e}")
        return ""

def migrate_vector_db(src_path: str, dest_path: str, backup_dir: str, force: bool = False, no_backup: bool = False) -> bool:
    """Migrate from one vector database to another"""
    # Check if destination exists
    if os.path.exists(dest_path) and not force:
        print(f"Destination {dest_path} already exists. Use --force to overwrite.")
        return False
    
    # Backup source if it exists and backup is not disabled
    if os.path.exists(src_path) and not no_backup:
        backup_path = backup_vector_database(src_path, backup_dir)
        if backup_path:
            print(f"Original database backed up to: {backup_path}")
    
    # Remove existing destination if force is enabled
    if os.path.exists(dest_path) and force:
        print(f"Removing existing directory at {dest_path}...")
        shutil.rmtree(dest_path)
    
    # Create empty destination directory
    os.makedirs(dest_path, exist_ok=True)
    print(f"Created directory at {dest_path}")
    
    # Create a flag file to indicate migration
    with open(os.path.join(dest_path, ".migrated"), "w") as f:
        f.write(f"Migration completed on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nMigration setup complete!")
    print("\nNext steps:")
    print("1. Run your application with --force_reload to rebuild the vector database")
    print("   Example: python main.py --force_reload")
    
    return True

def list_resources() -> None:
    """List resources that can be cleaned up"""
    resources = []
    
    # Check for FAISS vector database
    if os.path.exists("storage/index_db"):
        size = get_directory_size("storage/index_db")
        resources.append(("FAISS vector database", "storage/index_db", f"{size:.2f} MB"))
    
    # Check for legacy Chroma DB (for migration purposes)
    if os.path.exists("storage/chroma_db"):
        size = get_directory_size("storage/chroma_db")
        resources.append(("Legacy Chroma database", "storage/chroma_db", f"{size:.2f} MB"))
    
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
    
    # Check for backups
    if os.path.exists("storage/backups"):
        backups = os.listdir("storage/backups")
        if backups:
            total_size = get_directory_size("storage/backups")
            resources.append((f"Backups ({len(backups)})", "storage/backups", f"{total_size:.2f} MB"))
    
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

def ensure_dirs(dirs: List[str]) -> None:
    """Ensure directories exist"""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Ensured directory exists: {dir_path}")

def main():
    parser = argparse.ArgumentParser(description="Clean up and manage RAG system resources")
    parser.add_argument("--list", action="store_true", help="List resources that can be cleaned up")
    parser.add_argument("--delete-all", action="store_true", help="Delete all resources (vector DB, records, and PDFs)")
    parser.add_argument("--delete-db", action="store_true", help="Delete the vector database")
    parser.add_argument("--delete-records", action="store_true", help="Delete the processed PDFs record")
    parser.add_argument("--delete-pdfs", action="store_true", help="Delete all PDF files")
    parser.add_argument("--db-path", default="storage/index_db", help="Path to the vector database")
    parser.add_argument("--record-file", default="storage/processed_pdfs.json", help="Path to the processed PDFs record file")
    parser.add_argument("--pdf-dir", default="data", help="Directory containing PDF files")
    parser.add_argument("--force", action="store_true", help="Force deletion without confirmation")
    
    # Migration arguments
    parser.add_argument("--migrate", action="store_true", help="Migrate from one vector database to another")
    parser.add_argument("--source", default="storage/chroma_db", help="Source vector database path (for migration)")
    parser.add_argument("--destination", default="storage/index_db", help="Destination vector database path (for migration)")
    parser.add_argument("--backup-dir", default="storage/backups", help="Directory for backups")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backups during migration")
    
    args = parser.parse_args()
    
    # Ensure storage directories exist
    if args.migrate:
        ensure_dirs(["storage", "data", args.backup_dir])
    
    if args.list:
        list_resources()
        return
    
    if args.migrate:
        migrate_vector_db(args.source, args.destination, args.backup_dir, args.force, args.no_backup)
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