import pytest
from app.llm_service import LLMService
import json

@pytest.mark.asyncio
async def test_preference_extraction():
    llm_service = LLMService("your-api-key")
    
    # Test case 1: Basic house search
    input_1 = "I want a 3-bedroom house in San Francisco under $1.5M"
    response_1 = await llm_service.extract_preferences(input_1)
    
    assert isinstance(response_1, dict)
    assert response_1.get('location') == 'San Francisco'
    assert response_1.get('min_bedrooms') == 3
    assert response_1.get('max_price') <= 1500000
    
    # Test case 2: Price range and property type
    input_2 = "Looking for apartments between $300k and $600k in Chicago"
    response_2 = await llm_service.extract_preferences(input_2)
    
    assert isinstance(response_2, dict)
    assert response_2.get('location') == 'Chicago'
    assert response_2.get('property_type') == 'apartment'
    assert response_2.get('max_price') <= 600000
    
    # Test case 3: Error handling
    input_3 = ""  # Empty input
    response_3 = await llm_service.extract_preferences(input_3)
    
    assert isinstance(response_3, dict)
    assert 'max_price' in response_3  # Should return default values
    assert 'location' in response_3
    assert 'min_bedrooms' in response_3

@pytest.mark.asyncio
async def test_response_format():
    llm_service = LLMService("your-api-key")
    
    test_input = "3 bedroom house in Seattle under $800k with a garage"
    response = await llm_service.extract_preferences(test_input)
    
    required_fields = ['max_price', 'location', 'min_bedrooms', 'must_have_features']
    for field in required_fields:
        assert field in response, f"Missing required field: {field}"
    
    assert isinstance(response['max_price'], (int, float))
    assert isinstance(response['location'], str)
    assert isinstance(response['min_bedrooms'], int)
    assert isinstance(response['must_have_features'], list) 