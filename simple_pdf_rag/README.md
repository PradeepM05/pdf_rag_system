# Simple PDF RAG System

A simple Retrieval Augmented Generation (RAG) system that reads PDF files and allows you to ask questions about their content.

## Features

- Loads PDF files from a specified directory
- Tracks processed PDFs to avoid reprocessing unchanged files
- Extracts and processes text content
- Splits text into manageable chunks
- Creates embeddings for semantic search
- Stores embeddings in a ChromaDB vector database
- Retrieves relevant chunks based on queries
- Provides answers using either OpenAI LLM or a fallback method

## Setup

1. Install the required packages:
   ```bash
   pip install pypdf langchain langchain-community sentence-transformers chromadb python-dotenv openai
   ```

2. (Optional) Set up your OpenAI API key in the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. Add your PDF files to the `data` directory or specify a different directory with the `--pdf_dir` argument.

2. Run the application:
   ```bash
   python pdf_rag.py
   ```
   
   Or with a custom PDF directory:
   ```bash
   python pdf_rag.py --pdf_dir /path/to/your/pdfs
   ```
   
   To force reprocessing of all PDFs, ignoring the processed list:
   ```bash
   python pdf_rag.py --force_reload
   ```

3. Ask questions about the content of your PDFs when prompted.

## How It Works

1. The system keeps track of processed PDFs using MD5 hashes to avoid reprocessing unchanged files
2. When new or modified PDFs are detected, they are processed and added to the vector store
3. ChromaDB is used as a persistent vector store, so embeddings are saved between runs
4. When you ask a question, the system:
   - Retrieves the most relevant chunks from the vector store
   - Either uses the OpenAI API to generate an answer (if configured)
   - Or provides you with the most relevant text excerpts

## Example

```
RAG system initialized successfully!
You can now ask questions about the PDFs.
Type 'exit' to quit.

Question: What are the main topics covered in the document?

Answer:
...
```

## Notes

- If you don't set up an OpenAI API key, the system will use a fallback method that returns the most relevant text excerpts without generating an answer.
- For better results, use clear, specific questions related to the content of your PDFs.
- The system creates a file named `processed_pdfs.json` to track which PDFs have been processed. Do not delete this file unless you want to reprocess all PDFs. 