import asyncio
from pathlib import Path
import sys

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent.parent
sys.path.append(str(backend_dir))

from app.models import ChatMessage
from app.services.chat_storage import ChatStorageService
from datetime import datetime

async def test_conversation_context():
    # Initialize chat storage
    chat_storage = ChatStorageService()
    
    # Create a test session
    session_id = "test_session"
    session = await chat_storage.create_session(session_id)
    
    # Add some test messages
    messages = [
        ChatMessage(role="human", content="I'm looking for properties in Sydney", timestamp=datetime.now()),
        ChatMessage(role="assistant", content="Let me help you find properties in Sydney", timestamp=datetime.now()),
        ChatMessage(role="human", content="I prefer near the beach", timestamp=datetime.now()),
        ChatMessage(role="assistant", content="I'll search for beachside properties", timestamp=datetime.now()),
    ]
    
    # Save messages
    for msg in messages:
        await chat_storage.save_message(session_id, msg)
    
    # Get session and print messages
    session = await chat_storage.get_session(session_id)
    print("\nAll messages in session:")
    for msg in session.messages:
        print(f"Role: {msg.role}, Content: {msg.content}")
    
    # Get conversation context
    from app.Agent.agent import get_conversation_context
    context = await get_conversation_context(session_id)
    print("\nConversation context:")
    print(context)

if __name__ == "__main__":
    asyncio.run(test_conversation_context()) 