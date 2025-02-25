import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.llm_service import LLMService

async def test_llm_response():
    # Initialize LLM service
    llm_service = LLMService("your-api-key")  # API key is hardcoded in LLMService now
    
    # Test cases
    test_inputs = [
        "I want a 3-bedroom house in San Francisco under $1.5M",
        "Looking for a condo in NYC with 2 bedrooms and a maximum budget of $800k",
        "Show me apartments in Chicago with at least 2 bedrooms between $300k and $600k"
    ]
    
    print("\n=== Testing LLM Response ===\n")
    
    for input_text in test_inputs:
        print(f"Input: {input_text}")
        try:
            response = await llm_service.extract_preferences(input_text)
            print("Response:", response)
            print("\nValidating response fields:")
            print("- max_price:", response.get('max_price'))
            print("- location:", response.get('location'))
            print("- min_bedrooms:", response.get('min_bedrooms'))
            print("- property_type:", response.get('property_type'))
            print("- must_have_features:", response.get('must_have_features'))
        except Exception as e:
            print(f"Error: {e}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(test_llm_response()) 