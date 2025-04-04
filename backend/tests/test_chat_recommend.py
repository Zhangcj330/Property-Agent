import pytest
from fastapi.testclient import TestClient
import uuid
from datetime import datetime
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from google.api_core.exceptions import ResourceExhausted
from langchain_core.messages import HumanMessage, AIMessage

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.main import app

@pytest.fixture
def client():
    """Get a test client for the app."""
    return TestClient(app)

@pytest.fixture
def mock_llm():
    """Mock the LLM service to avoid API calls."""
    with patch('app.llm_service.LLMService.process_user_input') as mock:
        yield mock

def test_chat_recommend_basic_flow(client: TestClient, mock_llm):
    """Test the basic recommendation flow with clear preferences"""
    session_id = str(uuid.uuid4())
    
    # Mock the LLM response
    mock_llm.return_value = {
        "messages": [AIMessage(content="I understand you're looking for a house in Chatswood.")],
        "userpreferences": {},
        "propertysearchrequest": {
            "location": ["NSW-Chatswood-2067"],
            "max_price": 3500000,
            "min_bedrooms": 3,
            "property_type": ["House"]
        },
        "is_complete": True
    }
    
    # Initial message with clear preferences
    response = client.post(
        "/api/v1/chat",
        json={
            "session_id": session_id,
            "user_input": "I am looking for a house in Sydney Chatswood 2067, with 3 bedrooms, max price 3.5M",
            "preferences": {},
            "search_params": {}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "preferences" in data
    assert "search_params" in data
    
    # Verify search parameters were extracted
    search_params = data["search_params"]
    assert "location" in search_params
    assert "NSW-Chatswood-2067" in search_params["location"]
    assert search_params["max_price"] == 3500000
    assert search_params["min_bedrooms"] == 3
    assert "House" in search_params["property_type"]

def test_chat_recommend_vague_preferences(client: TestClient, mock_llm):
    """Test handling of vague preferences"""
    session_id = str(uuid.uuid4())
    
    # Mock the LLM response
    mock_llm.return_value = {
        "messages": [AIMessage(content="Could you please specify which area you're interested in?")],
        "userpreferences": {},
        "propertysearchrequest": {},
        "is_complete": False
    }
    
    response = client.post(
        "/api/v1/chat",
        json={
            "session_id": session_id,
            "user_input": "I want a nice house in a good area",
            "preferences": {},
            "search_params": {}
        }
    )
    assert response.status_code == 200
    data = response.json()
    
    # Verify that the system asks for clarification
    assert data["response"]
    assert "location" not in data["search_params"] or not data["search_params"]["location"]

def test_chat_recommend_conversation_flow(client: TestClient, mock_llm):
    """Test a multi-message conversation flow"""
    session_id = str(uuid.uuid4())
    
    # Mock first response
    mock_llm.return_value = {
        "messages": [AIMessage(content="I see you're interested in Sydney. Could you be more specific about which area?")],
        "userpreferences": {},
        "propertysearchrequest": {"location": ["NSW-Sydney"]},
        "is_complete": False
    }
    
    # First message
    response1 = client.post(
        "/api/v1/chat",
        json={
            "session_id": session_id,
            "user_input": "I'm looking for a property in Sydney",
            "preferences": {},
            "search_params": {}
        }
    )
    assert response1.status_code == 200
    
    # Mock second response
    mock_llm.return_value = {
        "messages": [AIMessage(content="I understand you're interested in the North Shore area with a budget of 2M.")],
        "userpreferences": {},
        "propertysearchrequest": {
            "location": ["NSW-North-Shore"],
            "max_price": 2000000
        },
        "is_complete": True
    }
    
    # Second message with more details
    response2 = client.post(
        "/api/v1/chat",
        json={
            "session_id": session_id,
            "user_input": "I prefer the North Shore area, my budget is around 2M",
            "preferences": response1.json()["preferences"],
            "search_params": response1.json()["search_params"]
        }
    )
    assert response2.status_code == 200
    data2 = response2.json()
    
    # Verify that preferences are being accumulated
    search_params = data2["search_params"]
    assert "location" in search_params
    assert any("NSW" in loc for loc in search_params["location"])
    assert search_params.get("max_price") == 2000000

def test_chat_recommend_error_handling(client: TestClient):
    """Test error handling scenarios"""
    
    # Test missing session_id
    response = client.post(
        "/api/v1/chat",
        json={
            "user_input": "Looking for a house"
        }
    )
    assert response.status_code == 200  # session_id is optional
    
    # Test empty message
    response = client.post(
        "/api/v1/chat",
        json={
            "session_id": str(uuid.uuid4()),
            "user_input": ""
        }
    )
    assert response.status_code == 422  # empty message is not allowed
    
    # Test invalid session_id format
    response = client.post(
        "/api/v1/chat",
        json={
            "session_id": "invalid-uuid",
            "user_input": "Looking for a house"
        }
    )
    assert response.status_code == 200  # we don't validate UUID format

def test_chat_recommend_preference_persistence(client: TestClient, mock_llm):
    """Test that preferences persist across multiple requests"""
    session_id = str(uuid.uuid4())
    
    # Mock first response
    mock_llm.return_value = {
        "messages": [AIMessage(content="I see you're interested in Chatswood.")],
        "userpreferences": {},
        "propertysearchrequest": {"location": ["NSW-Chatswood"]},
        "is_complete": False
    }
    
    # First message with location
    response1 = client.post(
        "/api/v1/chat",
        json={
            "session_id": session_id,
            "user_input": "I want to buy in Chatswood",
            "preferences": {},
            "search_params": {}
        }
    )
    assert response1.status_code == 200
    
    # Mock second response
    mock_llm.return_value = {
        "messages": [AIMessage(content="I understand your budget is 2M.")],
        "userpreferences": {},
        "propertysearchrequest": {
            "location": ["NSW-Chatswood"],
            "max_price": 2000000
        },
        "is_complete": True
    }
    
    # Second message with price
    response2 = client.post(
        "/api/v1/chat",
        json={
            "session_id": session_id,
            "user_input": "My budget is 2M",
            "preferences": response1.json()["preferences"],
            "search_params": response1.json()["search_params"]
        }
    )
    assert response2.status_code == 200
    data2 = response2.json()
    
    # Verify that both location and price are present
    search_params = data2["search_params"]
    assert "location" in search_params
    assert "NSW-Chatswood" in str(search_params["location"])
    assert search_params.get("max_price") == 2000000

def test_chat_recommend_contradictory_preferences(client: TestClient, mock_llm):
    """Test handling of contradictory preferences"""
    session_id = str(uuid.uuid4())
    
    # Mock the LLM response for contradictory input
    mock_llm.return_value = {
        "messages": [AIMessage(content="I notice there might be a contradiction in your requirements. You mentioned wanting both a quiet location and being in the heart of CBD with nightlife. Could you clarify which aspect is more important to you?")],
        "userpreferences": {},
        "propertysearchrequest": {},
        "is_complete": False
    }
    
    response = client.post(
        "/api/v1/chat",
        json={
            "session_id": session_id,
            "user_input": "I want a quiet house in the heart of the CBD with lots of nightlife",
            "preferences": {},
            "search_params": {}
        }
    )
    assert response.status_code == 200
    data = response.json()
    
    # Verify that the system identifies the contradiction
    assert data["response"]
    assert "contradiction" in data["response"].lower() or \
           "clarify" in data["response"].lower() 