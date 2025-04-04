import pytest
import pytest_asyncio
import uuid
from datetime import datetime
from app.services.chat_storage import ChatStorageService
from app.models import ChatMessage, ChatSession

@pytest_asyncio.fixture
async def chat_storage():
    service = ChatStorageService()
    yield service
    # 清理测试数据的逻辑可以在这里添加

@pytest.mark.asyncio
async def test_create_and_get_session(chat_storage):
    # 创建会话
    session = await chat_storage.create_session()
    assert session.session_id is not None
    assert isinstance(session.created_at, datetime)
    assert isinstance(session.last_active, datetime)
    assert len(session.messages) == 0

    # 获取会话
    retrieved_session = await chat_storage.get_session(session.session_id)
    assert retrieved_session is not None
    assert retrieved_session.session_id == session.session_id
    assert isinstance(retrieved_session.created_at, datetime)
    assert isinstance(retrieved_session.last_active, datetime)

@pytest.mark.asyncio
async def test_save_and_retrieve_message(chat_storage):
    # 创建会话
    session = await chat_storage.create_session()
    
    # 创建测试消息
    test_message = ChatMessage(
        role="user",
        content="Test message",
        timestamp=datetime.now()
    )
    
    # 保存消息
    await chat_storage.save_message(session.session_id, test_message)
    
    # 获取更新后的会话
    updated_session = await chat_storage.get_session(session.session_id)
    assert len(updated_session.messages) == 1
    saved_message = updated_session.messages[0]
    assert saved_message.role == test_message.role
    assert saved_message.content == test_message.content
    assert isinstance(saved_message.timestamp, datetime)

@pytest.mark.asyncio
async def test_update_session_state(chat_storage):
    # 创建会话
    session = await chat_storage.create_session()
    
    # 测试数据
    test_preferences = {"language": "zh", "model": "gpt-4"}
    test_search_params = {"query": "test", "filters": {"category": "tech"}}
    
    # 更新会话状态
    await chat_storage.update_session_state(
        session.session_id,
        preferences=test_preferences,
        search_params=test_search_params
    )
    
    # 验证更新
    updated_session = await chat_storage.get_session(session.session_id)
    assert updated_session.preferences == test_preferences
    assert updated_session.search_params == test_search_params

@pytest.mark.asyncio
async def test_clear_session(chat_storage):
    # 创建会话
    session = await chat_storage.create_session()
    
    # 清除会话
    await chat_storage.clear_session(session.session_id)
    
    # 验证会话已被删除
    deleted_session = await chat_storage.get_session(session.session_id)
    assert deleted_session is None

@pytest.mark.asyncio
async def test_session_with_custom_id(chat_storage):
    # 使用自定义 ID 创建会话
    custom_id = f"test-{uuid.uuid4()}"
    session = await chat_storage.create_session(session_id=custom_id)
    assert session.session_id == custom_id
    
    # 验证可以获取到会话
    retrieved_session = await chat_storage.get_session(custom_id)
    assert retrieved_session is not None
    assert retrieved_session.session_id == custom_id 