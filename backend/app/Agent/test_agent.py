import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent.parent
sys.path.append(str(backend_dir))

import pytest
from datetime import datetime
from langchain_core.messages import HumanMessage
from .agent import agent, extract_preferences, get_session_state, chat_storage

@pytest.mark.asyncio
async def test_session_id_propagation():
    """Test session ID propagation through the agent workflow"""
    
    # 设置测试数据
    test_session_id = "test_session_123"
    test_message = "I'm looking for a modern house in Melbourne with 4 bedrooms"
    
    
    print(f"\n=== Testing Session ID Propagation ===")
    print(f"Initial session_id: {test_session_id}")
    
    # 创建会话
    session = await chat_storage.create_session(test_session_id)
    print(f"\nCreated session: {session}")
    
    # 测试 extract_preferences 工具
    print("\nTesting direct tool call:")
    direct_result = await extract_preferences.ainvoke({
        "session_id": test_session_id,
        "user_message": test_message
    })
    print(f"Direct tool call result received")
    print(f"Direct result: {direct_result}")
    
    # 验证提取的偏好
    preferences = direct_result["preferences"]
    assert len(preferences) > 0, "Should extract at least one preference"
    
    # 验证 Style 偏好
    style_prefs = [p for p in preferences if p["category"] == "Style"]
    assert any("modern" in p["value"].lower() for p in style_prefs), "Should extract modern style preference"
    
    # 验证搜索参数
    search_params = direct_result["search_params"]
    assert len(search_params) > 0, "Should extract at least one search parameter"
    
    # 验证位置参数
    location_params = [p for p in search_params if p["param_name"] == "location"]
    assert any("melbourne" in str(p["value"]).lower() for p in location_params), "Should extract Melbourne location"
    
    # 验证卧室数量
    bedroom_params = [p for p in search_params if p["param_name"] == "min_bedrooms"]
    assert any(p["value"] == 4 for p in bedroom_params), "Should extract 4 bedrooms requirement"

        