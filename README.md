# PDF RAG System

A Retrieval Augmented Generation (RAG) system for PDF documents that allows you to ask questions about the content of your PDF files. The system implements document versioning to properly handle updates to your PDF documents.

## Features

- Load PDF files and extract text content
- Split text into chunks for effective processing
- Create vector embeddings using HuggingFace models
- Store embeddings in ChromaDB with document version tracking
- Mark old versions as "deprecated" when PDFs are updated
- Query only current versions of documents
- Generate answers using OpenAI API (with fallback for when API key is not available)
- Utilities for cleaning up system resources

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/PradeepM05/pdf_rag_system.git
   cd pdf_rag_system
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   Note: If you don't have an OpenAI API key, the system will still work in fallback mode, providing relevant excerpts instead of generated answers.

## Usage

1. Add your PDF files to the `data` directory (create it if it doesn't exist)
2. Run the application:
   ```bash
   python pdf_rag.py
   ```
3. Ask questions about the content of the PDFs
4. Type 'exit' to quit

### Cleanup Utility

The system includes a cleanup utility to manage resources:

```bash
python cleanup.py [options]
```

Options:
- `--list`: List resources that can be cleaned up
- `--delete-db`: Delete the vector database
- `--delete-records`: Delete the processed PDFs record
- `--delete-pdfs`: Delete all PDF files (will prompt for confirmation)
- `--delete-all`: Delete all resources (database, records, and PDFs)
- `--force`: Force deletion without confirmation prompts

## How it Works

1. **PDF Processing**: The system extracts text from PDF files and splits it into manageable chunks.

2. **Document Versioning**: 
   - Each PDF file is tracked using its MD5 hash
   - When a file is modified, old document chunks are marked as "deprecated"
   - New chunks are added with a "current" status
   - The system only retrieves "current" chunks when answering questions

3. **Vector Store**:
   - ChromaDB is used as a persistent vector store
   - Document chunks are stored with metadata about their source, version, and status

4. **Retrieval**:
   - When asking a question, the system finds the most relevant document chunks marked as "current"
   - If an OpenAI API key is available, it uses these chunks to generate a coherent answer
   - Otherwise, it returns the relevant excerpts directly

## License

MIT 