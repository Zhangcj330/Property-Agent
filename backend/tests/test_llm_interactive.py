import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.llm_service import LLMService

async def test_interactive_llm():
    llm_service = LLMService()
    
    # Test cases with varying completeness
    test_cases = [
        "I want to buy a house",  # Missing most information
        "Looking for a house in Seattle",  # Missing price and bedrooms
        "I need a 3-bedroom house",  # Missing location and price
        "I want a 3-bedroom house in San Francisco under $1.5M",  # Complete information
    ]
    
    print("\n=== Testing Interactive LLM Response ===\n")
    
    for input_text in test_cases:
        print(f"User: {input_text}")
        response, preferences = await llm_service.process_user_input(input_text)
        print(f"Assistant: {response}")
        print(f"Extracted Preferences: {preferences}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(test_interactive_llm()) 