import pytest
from app.services.preference_service import PreferenceService, extract_preferences_and_search_params
from app.models import ChatMessage, ChatSession
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage

@pytest.mark.asyncio
async def test_llm_response_debug():
    """Test to debug LLM response"""
    service = PreferenceService()
    
    # Test message
    context = "Previous conversation: User is looking for a property"
    recent_message = "I prefer a quiet neighborhood with good schools nearby"
    current_preferences = {}
    current_search_params = {}
    
    # Call _extract_with_llm directly to see the raw response
    preferences, search_params = await service._extract_with_llm(
        context,
        recent_message,
        current_preferences,
        current_search_params
    )
    
    # Print raw results
    print("\nLLM Debug Information:")
    print(f"Input Context: {context}")
    print(f"Input Message: {recent_message}")
    print(f"Raw Preferences: {preferences}")
    print(f"Raw Search Params: {search_params}")
    
    # Add assertions to verify LLM response format
    assert preferences is not None, "Preferences should not be None"
    assert search_params is not None, "Search params should not be None"
    
    if preferences:
        print("\nPreference Details:")
        for pref in preferences:
            print(f"- Type: {pref.preference_type}")
            print(f"  Category: {pref.category}")
            print(f"  Value: {pref.value}")
            print(f"  Importance: {pref.importance}")
            print(f"  Reason: {pref.reason}")
            print()

def create_mock_session():
    """Helper function to create a mock session"""
    session_id = "test_session_123"
    messages = [
        ChatMessage(
            role="user",
            content="I'm looking for a modern house in Chatswood, budget around 1.5 million",
            timestamp=datetime.now()
        ),
        ChatMessage(
            role="assistant",
            content="I understand you're looking for a modern house in Chatswood with a budget of 1.5 million.",
            timestamp=datetime.now()
        )
    ]
    
    return ChatSession(
        session_id=session_id,
        messages=messages,
        preferences={},
        search_params={}
    )

@pytest.mark.asyncio
async def test_extract_from_context():
    # Initialize service
    service = PreferenceService()
    
    # Test extraction with real LLM
    recent_message = "I prefer a quiet neighborhood with good schools nearby"
    preferences, search_params = await service.extract_from_context("test_session", recent_message)
    
    # Print debug information
    print("\nDebug Information for extract_from_context:")
    print(f"Recent Message: {recent_message}")
    print(f"Extracted Preferences: {preferences}")
    print(f"Extracted Search Params: {search_params}")
    
    # Assertions
    assert preferences is not None, "Preferences should not be None"
    assert search_params is not None, "Search params should not be None"
    
    if preferences:
        pref = preferences[0]
        print(f"\nFirst Preference Details:")
        print(f"Type: {pref.preference_type}")
        print(f"Category: {pref.category}")
        print(f"Value: {pref.value}")
        print(f"Importance: {pref.importance}")
        print(f"Reason: {pref.reason}")

@pytest.mark.asyncio
async def test_extract_preferences_and_search_params():
    # Test data
    session_id = "test_session"
    user_message = "I prefer a quiet neighborhood with good schools nearby"
    
    # Call the function directly
    result = await extract_preferences_and_search_params(session_id, user_message)
    
    # Print debug information
    print("\nDebug Information for extract_preferences_and_search_params:")
    print(f"Input Message: {user_message}")
    print(f"Result: {result}")
    
    # Assertions
    assert result is not None, "Result should not be None"
    assert "preferences" in result, "Result should contain preferences"
    assert "search_params" in result, "Result should contain search_params"
    assert "updated_preferences" in result, "Result should contain updated_preferences"
    assert "updated_search_params" in result, "Result should contain updated_search_params"

@pytest.mark.asyncio
async def test_extract_with_llm_detailed():
    """Detailed test of _extract_with_llm functionality"""
    service = PreferenceService()
    
    # Test inputs
    context = """Recent Dialogue:
--- Round 1 ---
User: I'm looking for a property in Sydney
Assistant: I'll help you find a property in Sydney. What are your preferences?

--- Round 2 ---
User: I prefer modern style homes with a budget around 2.5M
Assistant: I understand you're looking for modern homes in Sydney with a budget of 2.5M."""
    
    recent_message = "I want it to be in a quiet area with good schools nearby"
    current_preferences = {
        "Style": {
            "preference": "modern",
            "importance": 0.5
        }
    }
    current_search_params = {
        "max_price": 2500000
    }
    
    try:
        print("\nStep 1: Attempting preference extraction...")
        preferences, search_params = await service._extract_with_llm(
            context,
            recent_message,
            current_preferences,
            current_search_params
        )
        
        print("\nStep 2: Extraction Results")
        print("Preferences:")
        if preferences:
            for i, pref in enumerate(preferences, 1):
                print(f"\nPreference {i}:")
                print(f"  Type: {pref.preference_type}")
                print(f"  Category: {pref.category}")
                print(f"  Value: {pref.value}")
                print(f"  Importance: {pref.importance}")
                print(f"  Reason: {pref.reason}")
        else:
            print("  No preferences extracted")
            
        print("\nSearch Parameters:")
        if search_params:
            for i, param in enumerate(search_params, 1):
                print(f"\nParameter {i}:")
                print(f"  Name: {param.param_name}")
                print(f"  Value: {param.value}")
                print(f"  Reason: {param.reason}")
        else:
            print("  No search parameters extracted")
        
        # Assertions
        assert preferences is not None, "Preferences should not be None"
        assert search_params is not None, "Search parameters should not be None"
        assert len(preferences) > 0, "Should have extracted at least one preference"
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        raise

@pytest.mark.asyncio
async def test_preference_extraction_chain():
    """Test the complete preference extraction chain"""
    # Initialize service
    service = PreferenceService()
    
    # Create a test session with messages
    session_id = "test_session_chain"
    messages = [
        ChatMessage(
            role="user",
            content="I'm looking for a property in Sydney",
            timestamp=datetime.now()
        ),
        ChatMessage(
            role="assistant",
            content="I'll help you find a property in Sydney. What are your preferences?",
            timestamp=datetime.now()
        ),
        ChatMessage(
            role="user",
            content="I prefer modern style homes with a budget around 2.5M",
            timestamp=datetime.now()
        ),
        ChatMessage(
            role="assistant",
            content="I understand you're looking for modern homes in Sydney with a budget of 2.5M.",
            timestamp=datetime.now()
        )
    ]
    
    # Create session
    session = ChatSession(
        session_id=session_id,
        messages=messages,
        preferences={},
        search_params={}
    )
    
    # Mock get_session
    async def mock_get_session(sid):
        if sid == session_id:
            return session
        return None
    service.chat_storage.get_session = mock_get_session
    
    try:
        # Step 1: Test extract_from_context
        print("\nStep 1: Testing extract_from_context")
        recent_message = "I want it to be in a quiet area with good schools nearby"
        preferences, search_params = await service.extract_from_context(session_id, recent_message)
        
        print("Extract from context results:")
        print(f"Preferences: {preferences}")
        print(f"Search params: {search_params}")
        
        # Step 2: Test extract_preferences_and_search_params with the same service instance
        print("\nStep 2: Testing extract_preferences_and_search_params")
        result = await extract_preferences_and_search_params(session_id, recent_message, service=service)
        
        print("Final results:")
        print(f"Result: {result}")
        
        # Assertions
        assert preferences is not None, "Preferences should not be None from extract_from_context"
        assert search_params is not None, "Search params should not be None from extract_from_context"
        assert len(preferences) > 0, "Should have extracted at least one preference"
        assert result["preferences"] == [pref.dict() for pref in preferences], "Preferences should match between both calls"
        assert result["search_params"] == [param.dict() for param in search_params], "Search params should match between both calls"
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        raise 