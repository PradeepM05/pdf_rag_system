import os
import argparse
from dotenv import load_dotenv

from src.retrieval.rag_system import RAGSystem
from src.llm.openai_integration import test_openai_connectivity
from src.config import config

# Load environment variables from .env file if it exists
load_dotenv("config/.env")

# Print debugging information about the OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY", "")

if api_key:
    print(f"OpenAI API key found")
else:
    print("WARNING: No OpenAI API key found in environment variables")

def main():
    parser = argparse.ArgumentParser(description="Simple PDF RAG System")
    parser.add_argument("--pdf_dir", default=config.DATA_DIR, help="Directory containing PDF files")
    parser.add_argument("--force_reload", action="store_true", help="Force reload of all PDFs, ignoring the processed list")
    parser.add_argument("--test_openai", action="store_true", help="Test OpenAI connectivity")
    parser.add_argument("--no_chat_history", action="store_true", help="Disable chat history (treat each question independently)")
    args = parser.parse_args()
    
    # Override chat history setting if specified
    if args.no_chat_history:
        config.ENABLE_CHAT_HISTORY = False
    
    # Test OpenAI connectivity if requested
    if args.test_openai:
        test_openai_connectivity()
    
    rag_system = RAGSystem(args.pdf_dir)
    
    if not rag_system.initialize(force_reload=args.force_reload):
        print("Failed to initialize RAG system.")
        return
    
    print("\nRAG system initialized successfully!")
    print("You can now ask questions about the PDFs.")
    print("Type 'exit' to quit, or 'clear' to clear chat history.")
    
    while True:
        question = input("\nQuestion: ")
        
        if question.lower() in ('exit', 'quit'):
            break
        elif question.lower() == 'clear':
            rag_system.clear_chat_history()
            continue
            
        answer = rag_system.query(question)
        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    main()