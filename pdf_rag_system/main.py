import os
import argparse
from dotenv import load_dotenv

from src.retrieval.rag_system import RAGSystem
from src.llm.openai_integration import test_openai_connectivity
from src.config import config  # Import the centralized configuration


# Load environment variables from .env file if it exists
load_dotenv("config/.env")

# Print debugging information about the OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY", "")

if api_key:
    print(f"OpenAI API key found") #: {api_key[:5]}...{api_key[-4:]}")
else:
    print("WARNING: No OpenAI API key found in environment variables")

def main():
    parser = argparse.ArgumentParser(description="Simple PDF RAG System")
    parser.add_argument("--pdf_dir", default=config.DATA_DIR, help="Directory containing PDF files")
    parser.add_argument("--force_reload", action="store_true", help="Force reload of all PDFs, ignoring the processed list")
    parser.add_argument("--test_openai", action="store_true", help="Test OpenAI connectivity")
    args = parser.parse_args()
    
    # Test OpenAI connectivity if requested
    if args.test_openai:
        test_openai_connectivity()
    
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