import os

# Define global variable for OpenAI availability
OPENAI_AVAILABLE = False

# Try to import OpenAI for completions
try:
    from llama_index.llms.openai import OpenAI
    OPENAI_AVAILABLE = True
    print("Successfully imported OpenAI from LlamaIndex")
except ImportError as e:
    print(f"OpenAI package not available: {e}")
    print("Using fallback for completions.")

def test_openai_connectivity():
    """Test OpenAI connectivity"""
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        try:
            test_llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
            response = test_llm.complete("This is a test message. Reply with 'OpenAI connection successful.'")
            print(f"OpenAI Test Result: {response}")
            print("OpenAI connectivity test completed successfully.")
            return True
        except Exception as e:
            print(f"OpenAI test failed: {e}")
            return False
    else:
        print("OpenAI packages not available or API key not set. Cannot test.")
        return False