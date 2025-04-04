import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent.parent
sys.path.append(str(backend_dir))

import pytest
import pytest_asyncio
from datetime import datetime
from typing import Dict, List

from app.services.preference_service import (
    PreferenceService,
    PreferenceUpdate,
    SearchParamUpdate,
    extract_preferences_and_search_params,
    get_current_preferences_and_search_params,
    infer_preference_from_rejection
)
from app.models import ChatMessage, ChatSession
from app.services.chat_storage import ChatStorageService

@pytest_asyncio.fixture
async def preference_service():
    """Fixture for PreferenceService instance"""
    service = PreferenceService()
    yield service

@pytest_asyncio.fixture
async def chat_session(preference_service):
    """Fixture for test chat session with initial messages"""
    session_id = "test-preference-session"
    chat_storage = ChatStorageService()
    session = await chat_storage.create_session(session_id)
    
    # Add test dialogue
    messages = [
        ChatMessage(
            role="user",
            content="I'm looking for a modern house in Chatswood, budget around 1.5-2M",
            timestamp=datetime.now()
        ),
        ChatMessage(
            role="assistant",
            content="I'll help you find modern houses in Chatswood area.",
            timestamp=datetime.now()
        ),
        ChatMessage(
            role="user",
            content="Preferably close to the train station for easy commute",
            timestamp=datetime.now()
        ),
        ChatMessage(
            role="assistant",
            content="I'll focus on properties near Chatswood station.",
            timestamp=datetime.now()
        )
    ]
    
    for message in messages:
        await chat_storage.save_message(session_id, message)
    
    yield session_id
    
    # Cleanup test data
    await chat_storage.clear_session(session_id)

@pytest.mark.asyncio
async def test_extract_from_context(preference_service, chat_session):
    """Test extracting preferences and search parameters from conversation context"""
    # Test message
    recent_message = "I'd like a house with a garden, preferably with three bedrooms"
    
    # Extract preferences and search parameters
    preferences, search_params = await preference_service.extract_from_context(
        chat_session,
        recent_message
    )
    
    # Verify extracted preferences
    assert len(preferences) > 0, "Should extract at least one preference"
    assert any(
        p for p in preferences 
        if p.category == "Features" and "garden" in p.value.lower()
    ), "Should extract garden preference"
    
    # Verify extracted search parameters
    assert any(
        p for p in search_params 
        if p.param_name == "min_bedrooms" and p.value == 3
    ), "Should extract bedroom requirement"
    assert any(
        p for p in search_params 
        if p.param_name == "location" and "chatswood" in str(p.value).lower()
    ), "Should maintain location context"

@pytest.mark.asyncio
async def test_update_user_preferences(preference_service, chat_session):
    """Test updating user preferences"""
    # Create test preference updates
    preference_updates = [
        PreferenceUpdate(
            preference_type="explicit",
            category="Style",
            value="modern",
            importance=0.8,
            reason="User explicitly stated modern style preference"
        ),
        PreferenceUpdate(
            preference_type="implicit",
            category="Transport",
            value="near_station",
            importance=0.6,
            reason="User mentioned commute convenience"
        )
    ]
    
    # Update preferences
    updated_preferences = await preference_service.update_user_preferences(
        chat_session,
        preference_updates
    )
    
    # Verify updates
    assert "Style" in updated_preferences, "Style preference should be added"
    assert updated_preferences["Style"]["preference"] == "modern"
    assert updated_preferences["Style"]["importance"] == 0.8
    
    assert "Transport" in updated_preferences, "Transport preference should be added"
    assert updated_preferences["Transport"]["preference"] == "near_station"
    assert updated_preferences["Transport"]["importance"] == 0.6
    assert updated_preferences["Transport"].get("implicit") is True

@pytest.mark.asyncio
async def test_update_search_params(preference_service, chat_session):
    """Test updating search parameters"""
    # Create test search parameter updates
    search_param_updates = [
        SearchParamUpdate(
            param_name="location",
            value=["NSW-chatswood-2067"],
            reason="User specified Chatswood"
        ),
        SearchParamUpdate(
            param_name="min_price",
            value=1500000,
            reason="User mentioned budget range"
        ),
        SearchParamUpdate(
            param_name="max_price",
            value=2000000,
            reason="User mentioned budget range"
        )
    ]
    
    # Update search parameters
    updated_params = await preference_service.update_search_params(
        chat_session,
        search_param_updates
    )
    
    # Verify updates
    assert "location" in updated_params, "Location should be added"
    assert "NSW-chatswood-2067" in updated_params["location"]
    assert updated_params["min_price"] == 1500000, "Min price should be set"
    assert updated_params["max_price"] == 2000000, "Max price should be set"

@pytest.mark.asyncio
async def test_prepare_conversation_context(preference_service):
    """Test conversation context preparation"""
    messages = [
        ChatMessage(role="user", content="First message", timestamp=datetime.now()),
        ChatMessage(role="assistant", content="First response", timestamp=datetime.now()),
        ChatMessage(role="system", content="System message", timestamp=datetime.now()),
        ChatMessage(role="user", content="Second message", timestamp=datetime.now()),
        ChatMessage(role="assistant", content="Second response", timestamp=datetime.now())
    ]
    
    context = preference_service._prepare_conversation_context(messages)
    
    # Verify context format
    assert "Round 1" in context, "Should include round numbering"
    assert "Round 2" in context, "Should include multiple rounds"
    assert "First message" in context, "Should include user messages"
    assert "First response" in context, "Should include assistant responses"
    assert "System message" not in context, "Should filter out system messages"

@pytest.mark.asyncio
async def test_extract_with_llm(preference_service):
    """Test LLM-based preference extraction"""
    context = """Recent Dialogue:
User: I'm looking for a house in Chatswood, budget under 2M
Assistant: I'll help you search for that."""
    recent_message = "I want a modern style house with three bedrooms"
    current_preferences = {}
    current_search_params = {}
    
    preferences, search_params = await preference_service._extract_with_llm(
        context,
        recent_message,
        current_preferences,
        current_search_params
    )
    
    # Verify LLM extraction results
    assert len(preferences) > 0, "Should extract at least one preference"
    assert len(search_params) > 0, "Should extract at least one search parameter"
    
    # Check specific extractions
    style_prefs = [p for p in preferences if p.category == "Style"]
    assert any("modern" in p.value.lower() for p in style_prefs), "Should extract modern style preference"
    
    bedroom_params = [p for p in search_params if p.param_name == "min_bedrooms"]
    assert any(p.value == 3 for p in bedroom_params), "Should extract bedroom requirement"

@pytest.mark.asyncio
async def test_infer_preference_from_rejection():
    """Test preference inference from property rejection"""
    session_id = "test-rejection-session"
    rejection_message = "This house is too noisy and the style is too old-fashioned"
    property_details = {
        "location": "NSW-chatswood-2067",
        "style": "traditional",
        "environment": "near main road",
        "price": 1800000
    }
    
    result = await infer_preference_from_rejection(
        session_id,
        rejection_message,
        property_details
    )
    # Verify inferred preferences
    preferences = result["preferences"]
    assert len(preferences) > 0, "Should infer at least one preference"
    
    # Check specific inferences
    env_prefs = [p for p in preferences if p["category"] == "Environment"]
    assert any("quiet" in p["value"].lower() for p in env_prefs), "Should infer quiet environment preference"
    
    style_prefs = [p for p in preferences if p["category"] == "Style"]
    assert any("modern" in p["value"].lower() for p in style_prefs), "Should infer modern style preference"

@pytest.mark.asyncio
async def test_location_ambiguity_handling(preference_service, chat_session):
    """Test handling of ambiguous location references"""
    recent_message = "I want to find a house in North Shore"
    
    preferences, search_params = await preference_service.extract_from_context(
        chat_session,
        recent_message
    )
    
    # Verify location parameter handling
    location_params = [p for p in search_params if p.param_name == "location"]
    assert len(location_params) > 0, "Should extract location parameter"
    
    # Check ambiguity handling
    location_value = location_params[0].value
    if isinstance(location_value, dict):
        assert "suggestions" in location_value, "Should provide area suggestions"
        assert len(location_value["suggestions"]) > 0, "Should have at least one suggestion"
        assert "context" in location_value, "Should provide context for suggestions"

@pytest.mark.asyncio
async def test_preference_importance_levels(preference_service, chat_session):
    """Test preference importance level detection"""
    test_messages = [
        ("I must have a three-bedroom house", 0.8),  # High importance
        ("I prefer modern style", 0.5),              # Medium importance
        ("A garden would be nice", 0.3),             # Low importance
    ]
    
    for message, expected_importance in test_messages:
        preferences, _ = await preference_service.extract_from_context(
            chat_session,
            message
        )
        
        assert len(preferences) > 0, f"Should extract preference from: {message}"
        
        if "must" in message.lower():
            assert any(0.7 <= p.importance <= 1.0 for p in preferences), "Should detect high importance"
        elif "prefer" in message.lower():
            assert any(0.4 <= p.importance <= 0.6 for p in preferences), "Should detect medium importance"
        elif "would be" in message.lower():
            assert any(0.1 <= p.importance <= 0.3 for p in preferences), "Should detect low importance"

@pytest.mark.asyncio
async def test_tool_functions():
    """Test utility tool functions"""
    session_id = "test-tool-functions"
    user_message = "I want a modern house in Chatswood, budget around 1.5-2M"
    
    # Test extraction function
    result = await extract_preferences_and_search_params(session_id, user_message)
    assert "preferences" in result, "Should include preferences"
    assert "search_params" in result, "Should include search parameters"
    assert "updated_preferences" in result, "Should include updated preferences"
    assert "updated_search_params" in result, "Should include updated search parameters"
    
    # Test current state retrieval
    current_state = await get_current_preferences_and_search_params(session_id)
    assert "preferences" in current_state, "Should include current preferences"
    assert "search_params" in current_state, "Should include current search parameters" 