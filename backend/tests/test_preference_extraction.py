import pytest
import asyncio
from unittest.mock import patch, MagicMock
import json
from typing import Dict, List
import sys
import os
import time  # 导入time模块用于添加延迟

# 添加backend目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.llm_service import LLMService
from app.models import UserPreferences, PropertySearchRequest
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.config import settings

# Mock LLM for simulating user responses
mock_user_llm = ChatOpenAI(
    api_key=settings.GEMINI_API_KEY,
    base_url=settings.BASE_URL,
    model="gemini-1.5-flash"
)

class TestPreferenceExtraction:
    """Test the extraction of user preferences through simulated conversations"""
    
    @pytest.fixture
    def llm_service(self):
        """Create a fresh LLM service for each test"""
        service = LLMService()
        # Clear any existing chat history
        service.chat_history = []
        return service
    
    async def simulate_conversation_until_complete(self, llm_service: LLMService, user_persona: str, 
                                   initial_query: str, max_turns: int = 5, delay_seconds: int = 5):
        """
        Simulate a conversation between the system and a user with a specific persona
        until all required information is collected or max_turns is reached
        
        Args:
            llm_service: The LLM service to test
            user_persona: Description of the user's personality and preferences
            initial_query: The user's initial query
            max_turns: Maximum number of conversation turns to simulate
            delay_seconds: Number of seconds to wait between conversation turns
        
        Returns:
            final_state: The final state after the conversation
            turns_taken: Number of conversation turns that occurred
        """
        # Process initial query
        print(f"\n初始查询: {initial_query}")
        state = await llm_service.process_user_input(initial_query)
        turns_taken = 1
         # Process the user response
        print(f"\n处理用户回复...")
        assistant_message = llm_service.chat_history[-1]["content"]
        print(f"\n对话轮次 {turns_taken}:")
        print(f"房产顾问: {assistant_message}")
        print(f"用户: {initial_query}")
        print(f"模糊字段: {state['ambiguities']}")
        print(f"用户偏好: {state['userpreferences']}")
        print("---")
        # Continue conversation until complete or max turns reached
        while not state.get("is_complete", False) and turns_taken < max_turns:
            # If there's no assistant message, break
            if not llm_service.chat_history or llm_service.chat_history[-1]["role"] != "assistant":
                break
                
            # Get the last assistant message
            assistant_message = llm_service.chat_history[-1]["content"]
            
            # Add delay between turns to simulate real conversation and avoid rate limits
            print(f"\n等待 {delay_seconds} 秒...")
            time.sleep(delay_seconds)
            
            # Generate user response based on persona and assistant message
            print(f"\n生成用户回复...")
            user_response = mock_user_llm.invoke([
                SystemMessage(content=f"""You are simulating a real estate buyer with the following persona:
                {user_persona}
                
                Respond naturally to the real estate agent's question below. Stay in character and provide information
                that aligns with your persona. Be conversational and realistic - don't be too direct or perfect in your answers.
                Sometimes be a bit vague or mention contradictory preferences as real humans do.
                
                If asked about specific details like price range, location, or property features, provide an answer
                that matches your persona's preferences, even if it's somewhat vague or contradictory.
                
                Don't mention that you're an AI or that this is a simulation."""),
                HumanMessage(content=f"Real estate agent: {assistant_message}")
            ])
            
            # Process the user response
            print(f"\n处理用户回复...")
            state = await llm_service.process_user_input(user_response.content)
            turns_taken += 1
            
            print(f"\n对话轮次 {turns_taken}:")
            print(f"房产顾问: {assistant_message}")
            print(f"用户: {user_response.content}")
            print(f"模糊字段: {state['ambiguities']}")
            print(f"用户偏好: {state['userpreferences']}")
            print("---")
                
        return state, turns_taken
    
    @pytest.mark.asyncio
    async def test_clear_preferences_extraction(self, llm_service):
        """Test extraction with a user who has clear preferences"""
        
        print("\n\n===== 开始测试：明确偏好提取 =====")
        
        user_persona = """
        You are Sarah, a 35-year-old marketing executive with a family of four (two children aged 5 and 7).
        Your preferences:
        - Looking for a house in Sydney's North Shore area
        - Budget is strictly between $2.5M and $3M
        - Need at least 4 bedrooms and 2 bathrooms
        - Must be in a good school district
        - Prefer modern style with open floor plan
        - Want a garden for the kids to play
        - Need parking for 2 cars
        - Commute to CBD should be under 45 minutes
        - Safe & Family-Friendly community

        You are decisive and clear about your requirements.
        """
        
        initial_query = "Hi, I'm looking for a family home in Sydney's North Shore. We have a budget of around $2.5-3M and need at least 4 bedrooms."
        
        final_state, turns = await self.simulate_conversation_until_complete(
            llm_service, 
            user_persona, 
            initial_query
        )
        
        # Check if key preferences were extracted correctly
        preferences = final_state["userpreferences"]
        search_params = final_state["propertysearchrequest"]
        
        assert final_state.get("is_complete", False), f"Failed to complete preference extraction in {turns} turns"
        assert "NSW" in search_params.get("state", ""), "Failed to extract state"
        assert search_params.get("min_price", 0) >= 2500000, "Failed to extract minimum price"
        assert search_params.get("max_price", 0) <= 3000000, "Failed to extract maximum price"
        assert search_params.get("min_bedrooms", 0) >= 4, "Failed to extract bedroom requirement"
        assert "house" in search_params.get("property_type", "").lower(), "Failed to extract property type"
        
        # Check if detailed preferences were captured
        assert any("school" in str(pref).lower() for pref in preferences.values()), "Failed to capture school preference"
        assert any("garden" in str(pref).lower() for pref in preferences.values()), "Failed to capture garden preference"
        
        print("\n=== 明确偏好测试结果 ===")
        print(f"完成对话轮次: {turns}")
        print(f"提取的搜索参数: {json.dumps(search_params, indent=2)}")
        print(f"提取的用户偏好: {json.dumps(preferences, indent=2)}")
    
    @pytest.mark.asyncio
    async def test_vague_preferences_extraction(self, llm_service):
        """Test extraction with a user who has vague preferences"""
        
        print("\n\n===== 开始测试：模糊偏好提取 =====")
        
        user_persona = """
        You are Michael, a 28-year-old software developer who's buying his first property.
        Your preferences:
        - Not sure about location but somewhere with a "cool vibe" and good cafes
        - Budget is "affordable" but when pressed you'll say around $600K-800K
        - Thinking maybe an apartment or townhouse, not a full house
        - Would like "modern" and "low maintenance"
        - Need good internet connectivity
        - Want to be close to public transport
        - Like the idea of having some outdoor space but not essential
        
        You are indecisive and often use vague terms rather than specifics.
        When pressed for details, you'll eventually provide them but with some uncertainty.
        """
        
        initial_query = "Hey there, I'm looking to buy my first place. I want something modern with a cool vibe, maybe in an up-and-coming area? Not sure about my budget yet but I don't want to break the bank."
        
        final_state, turns = await self.simulate_conversation_until_complete(
            llm_service, 
            user_persona, 
            initial_query,
            max_turns=15,  # More turns allowed for vague users
            delay_seconds=5  # Slightly longer delay for complex scenarios
        )
        
        # Check if the system prompted for clarification on vague terms
        conversation = llm_service.chat_history
        clarification_requested = False
        for msg in conversation:
            if msg["role"] == "assistant" and any(term in msg["content"].lower() for term in ["budget", "price", "afford", "cost"]):
                clarification_requested = True
                break
                
        assert clarification_requested, "System failed to request clarification on vague budget"
        
        # Check if preferences were eventually extracted
        search_params = final_state["propertysearchrequest"]
        assert final_state.get("is_complete", False), f"Failed to complete preference extraction in {turns} turns"
        assert search_params.get("location") is not None, "Failed to extract any location"
        assert search_params.get("max_price") is not None, "Failed to extract any price range"
        assert search_params.get("property_type") is not None, "Failed to extract property type"
        
        print("\n=== 模糊偏好测试结果 ===")
        print(f"完成对话轮次: {turns}")
        print(f"提取的搜索参数: {json.dumps(search_params, indent=2)}")
        print(f"提取的用户偏好: {json.dumps(final_state['userpreferences'], indent=2)}")
    
    @pytest.mark.asyncio
    async def test_contradictory_preferences_extraction(self, llm_service):
        """Test extraction with a user who has contradictory preferences"""
        
        print("\n\n===== 开始测试：矛盾偏好提取 =====")
        
        user_persona = """
        You are Emma, a 42-year-old doctor looking for an investment property.
        Your preferences have some contradictions:
        - Want a "quiet, peaceful neighborhood" but also "walking distance to restaurants and nightlife"
        - Budget is "around $700K" but expect "high-end finishes" and "luxury features"
        - Want "low maintenance" but also "large garden"
        - Need it to be "close to the hospital" (in the city) but want "lots of space and privacy"
        - Want "character and charm" but also "modern amenities"
        
        You don't realize these preferences are somewhat contradictory.
        When asked to prioritize, you'll reluctantly choose:
        - Location near hospital over space/privacy
        - Modern amenities over character/charm
        - Budget is firm at $700K
        - Quiet neighborhood is more important than nightlife
        """
        
        initial_query = "I'm looking for an investment property that's peaceful and quiet but also close to restaurants and nightlife. My budget is around $700K and I want something with luxury features and high-end finishes. It should be low maintenance but have a nice garden."
        
        final_state, turns = await self.simulate_conversation_until_complete(
            llm_service, 
            user_persona, 
            initial_query,
            max_turns=15,
            delay_seconds=5  # Longer delay for complex scenarios
        )
        
        # Check if ambiguities were detected and resolved
        conversation = llm_service.chat_history
        contradiction_addressed = False
        for msg in conversation:
            if msg["role"] == "assistant" and any(term in msg["content"].lower() for term in ["clarify", "trade-off", "balance", "prioritize"]):
                contradiction_addressed = True
                break
                
        assert contradiction_addressed, "System failed to address contradictory preferences"
        
        # Check if preferences were eventually extracted
        search_params = final_state["propertysearchrequest"]
        assert final_state.get("is_complete", False), f"Failed to complete preference extraction in {turns} turns"
        assert search_params.get("location") is not None, "Failed to extract location"
        assert search_params.get("max_price") is not None, "Failed to extract price range"
        assert search_params.get("property_type") is not None, "Failed to extract property type"
        
        print("\n=== 矛盾偏好测试结果 ===")
        print(f"完成对话轮次: {turns}")
        print(f"提取的搜索参数: {json.dumps(search_params, indent=2)}")
        print(f"提取的用户偏好: {json.dumps(final_state['userpreferences'], indent=2)}")
        print(f"检测到的矛盾: {json.dumps(final_state.get('ambiguities', []), indent=2)}")
    
    @pytest.mark.asyncio
    async def test_changing_preferences_extraction(self, llm_service):
        """Test extraction with a user who changes preferences during conversation"""
        
        user_persona = """
        You are David, a 50-year-old empty nester looking to downsize.
        Your preferences change during the conversation:
        - Initially looking for a house, but then realize an apartment might be better
        - Start with a budget of $1M but then decide you can go up to $1.3M
        - First mention wanting to be in the suburbs, then shift to preferring the city
        - Initially want a garden but then decide maintenance is too much work
        - Change your mind about the number of bedrooms needed from 3 to 2
        
        You tend to change your mind as you think more about your requirements.
        When asked specific questions, you'll sometimes contradict your earlier statements
        and then acknowledge that your preferences have changed.
        """
        
        initial_query = "Hello, I'm looking to downsize from my family home. I think I want a smaller house in the suburbs with about 3 bedrooms and a nice garden. My budget is around $1 million."
        
        final_state, turns = await self.simulate_conversation_until_complete(
            llm_service, 
            user_persona, 
            initial_query,
            max_turns=15
        )
        
        # Check if preferences were updated correctly
        search_params = final_state["propertysearchrequest"]
        assert final_state.get("is_complete", False), f"Failed to complete preference extraction in {turns} turns"
        
        # We can't assert exact values since the changing preferences are simulated
        # and might vary, but we can check that key fields were extracted
        assert search_params.get("location") is not None, "Failed to extract location"
        assert search_params.get("max_price") is not None, "Failed to extract price range"
        assert search_params.get("property_type") is not None, "Failed to extract property type"
        
        print("\n=== Changing Preferences Test Results ===")
        print(f"Completed in {turns} turns")
        print(f"Extracted search parameters: {json.dumps(search_params, indent=2)}")
        print(f"Extracted preferences: {json.dumps(final_state['userpreferences'], indent=2)}")
        
    @pytest.mark.asyncio
    async def test_minimal_information_extraction(self, llm_service):
        """Test extraction with a user who provides minimal information"""
        
        user_persona = """
        You are Alex, a 32-year-old busy professional who doesn't have time for lengthy conversations.
        Your communication style:
        - You provide very brief responses, often just a few words
        - You don't volunteer information unless specifically asked
        - You're impatient and want quick results
        
        Your actual preferences (which you'll reveal only when directly asked):
        - Looking for an apartment in Melbourne CBD
        - Budget is exactly $850K, not a dollar more
        - Need 2 bedrooms minimum
        - Must have secure parking
        - Must be within walking distance to public transport
        - Prefer modern style
        """
        
        initial_query = "I need an apartment in Melbourne. What do you have?"
        
        final_state, turns = await self.simulate_conversation_until_complete(
            llm_service, 
            user_persona, 
            initial_query,
            max_turns=15
        )
        
        # Check if the system was able to extract preferences despite minimal information
        search_params = final_state["propertysearchrequest"]
        assert final_state.get("is_complete", False), f"Failed to complete preference extraction in {turns} turns"
        assert "Melbourne" in str(search_params.get("location", "")), "Failed to extract location"
        assert search_params.get("max_price") is not None, "Failed to extract price range"
        assert search_params.get("property_type") == "apartment", "Failed to extract property type"
        
        print("\n=== Minimal Information Test Results ===")
        print(f"Completed in {turns} turns")
        print(f"Extracted search parameters: {json.dumps(search_params, indent=2)}")
        print(f"Extracted preferences: {json.dumps(final_state['userpreferences'], indent=2)}")
        
    @pytest.mark.asyncio
    async def test_ambiguity_detection(self, llm_service):
        """Test the detection and resolution of ambiguities in user preferences"""
        
        print("\n\n===== 开始测试：矛盾检测能力 =====")
        
        user_persona = """
        You are Jordan, a 38-year-old tech executive with highly contradictory preferences.
        Your communication style:
        - You speak confidently but your requirements have significant internal contradictions
        - You don't realize your requirements conflict with each other
        - You become indecisive when the contradictions are pointed out
        
        Your contradictory preferences:
        - Want a "waterfront property" but your budget is only $600K (unrealistic in most Australian cities)
        - Want a "quiet countryside feel" but also "walking distance to CBD offices"
        - Want a "newly built modern home" but also "character-filled historic property"
        - Need "minimum 4 bedrooms" but want a "low-maintenance small property"
        - Want to be in an "exclusive prestigious suburb" but your budget is too low for such areas
        
        When contradictions are pointed out, you'll reluctantly prioritize:
        - Location over size (happy with smaller place in better location)
        - Modern over historic (functionality over character)
        - Closer to city over waterfront views
        """
        
        initial_query = "I'm looking for a waterfront property that's quiet and peaceful but within walking distance to CBD. My budget is around $600K and I need at least 4 bedrooms. I prefer both modern and historic features, with minimal maintenance."
        
        # Process the initial query
        state = await llm_service.process_user_input(initial_query)
        
        # Check if ambiguities were detected
        assert state.get("has_ambiguities", False), "Failed to detect obvious contradictions"
        assert len(state.get("ambiguities", [])) > 0, "No ambiguities were identified"
        
        # Verify that at least one contradiction is of high importance
        ambiguities = state.get("ambiguities", [])
        high_importance_ambiguities = [a for a in ambiguities if a.get("importance") == "high"]
        assert len(high_importance_ambiguities) > 0, "Failed to mark any contradiction as high importance"
        
        # Check that the detected ambiguities make sense
        ambiguity_fields = [a.get("field") for a in ambiguities]
        expected_fields = ["location", "price", "size", "property_type"]
        assert any(field in ambiguity_fields for field in expected_fields), "Failed to detect contradictions in expected fields"
        
        # Verify that clarification questions are generated for ambiguities
        assert all("question" in a or "clarification_question" in a for a in ambiguities), "Missing clarification questions for ambiguities"
        
        print("\n=== Ambiguity Detection Test Results ===")
        print(f"Number of ambiguities detected: {len(ambiguities)}")
        print(f"High importance ambiguities: {len(high_importance_ambiguities)}")
        print(f"Detected ambiguities: {json.dumps(ambiguities, indent=2)}")
        
        # Simulate answering one clarification
        if ambiguities and len(llm_service.chat_history) >= 2:
            clarification_question = llm_service.chat_history[-1]["content"]
            print(f"\nClarification question: {clarification_question}")
            
            # Generate user response to clarification
            user_response = mock_user_llm.invoke([
                SystemMessage(content=f"""You are simulating a person with the following persona:
                {user_persona}
                
                Respond to the real estate agent's question about contradictions in your requirements.
                Acknowledge the contradiction and reluctantly prioritize based on your persona."""),
                HumanMessage(content=f"Real estate agent: {clarification_question}")
            ])
            
            print(f"User response: {user_response.content}")
            
            # Process the clarification response
            final_state = await llm_service.process_user_input(user_response.content)
            
            # Check if ambiguities were resolved
            remaining_ambiguities = final_state.get("ambiguities", [])
            print(f"Remaining ambiguities after clarification: {len(remaining_ambiguities)}")
            
            # The number of ambiguities should decrease or their importance should change
            assert len(remaining_ambiguities) <= len(ambiguities), "Clarification didn't reduce ambiguities"
            
    @pytest.mark.asyncio
    async def test_simple_location_ambiguity(self, llm_service):
        """Test detection of vague location preferences"""
        
        print("\n\n===== 开始测试：位置模糊检测 =====")
        
        # 简单的初始查询，只提到城市而不是具体区域
        initial_query = "I'm looking for a house in Sydney. My budget is around $1.5 million."
        
        # 处理初始查询
        state = await llm_service.process_user_input(initial_query)
        
        # 检查是否检测到模糊性
        ambiguities = state.get("ambiguities", [])
        print(f"\n检测到的模糊性: {json.dumps(ambiguities, indent=2)}")
        
        # 检查是否有位置模糊性
        location_ambiguities = [a for a in ambiguities if a.get("field") == "location"]
        assert len(location_ambiguities) > 0, "未检测到位置模糊性"
        
        print("\n当前搜索参数:")
        print(json.dumps(state.get("propertysearchrequest", {}), indent=2))
        
        # 获取澄清问题或后续询问
        if len(llm_service.chat_history) >= 2:
            clarification_question = llm_service.chat_history[-1]["content"]
            print(f"\n澄清问题: {clarification_question}")
            
            # 模拟用户对澄清问题的回答
            user_response = "I'm interested in the Eastern Suburbs, maybe Bondi or Randwick."
            print(f"\n用户回答: {user_response}")
            
            # 处理澄清回答
            final_state = await llm_service.process_user_input(user_response)
            
            # 检查模糊性是否解决
            remaining_ambiguities = final_state.get("ambiguities", [])
            print(f"\n澄清后剩余模糊性: {len(remaining_ambiguities)}")
            
            # 检查提取的信息
            search_params = final_state["propertysearchrequest"]
            preferences = final_state["userpreferences"]
            
            print("\n最终搜索参数:")
            print(json.dumps(search_params, indent=2))
            print("\n最终用户偏好:")
            print(json.dumps(preferences, indent=2))
            
            # 检查位置信息是否更新 - 可能在位置字段或用户偏好中
            location_updated = False
            
            # 检查搜索参数中的位置
            location_str = str(search_params.get("location", "")).lower()
            if any(suburb in location_str for suburb in ["eastern", "bondi", "randwick"]):
                location_updated = True
            
            # 检查用户偏好中的位置
            if "Location" in preferences:
                location_pref = str(preferences["Location"][0]).lower()
                if any(suburb in location_pref for suburb in ["eastern", "bondi", "randwick"]):
                    location_updated = True
            
            assert location_updated, "澄清后未正确更新位置信息"

    @pytest.mark.asyncio
    async def test_simple_price_ambiguity(self, llm_service):
        """Test detection of vague price preferences"""
        
        print("\n\n===== 开始测试：价格模糊检测 =====")
        
        # 使用模糊的价格表述
        initial_query = "I need an affordable apartment in Melbourne that's modern and close to public transport."
        
        # 处理初始查询
        state = await llm_service.process_user_input(initial_query)
        
        # 检查是否检测到模糊性
        ambiguities = state.get("ambiguities", [])
        print(f"\n检测到的模糊性: {json.dumps(ambiguities, indent=2)}")
        
        # 获取澄清问题或后续询问
        if len(llm_service.chat_history) >= 2:
            agent_question = llm_service.chat_history[-1]["content"]
            print(f"\n房产顾问问题: {agent_question}")
            
            # 模拟用户具体的价格回答
            user_response = "My budget is around $500K to $650K maximum."
            print(f"\n用户回答: {user_response}")
            
            # 处理价格回答
            final_state = await llm_service.process_user_input(user_response)
            
            # 检查提取的价格信息是否正确
            search_params = final_state["propertysearchrequest"]
            print(f"\n提取的搜索参数: {json.dumps(search_params, indent=2)}")
            
            assert search_params.get("min_price") is not None, "未提取最低价格"
            assert search_params.get("max_price") is not None, "未提取最高价格"
            assert 450000 <= search_params.get("min_price", 0) <= 550000, "最低价格提取不准确"
            assert 600000 <= search_params.get("max_price", 0) <= 700000, "最高价格提取不准确"

    @pytest.mark.asyncio
    async def test_simple_contradictory_preferences(self, llm_service):
        """Test detection of simple contradictory preferences"""
        
        print("\n\n===== 开始测试：简单矛盾偏好检测 =====")
        
        # 包含明显矛盾的初始查询
        initial_query = "I want a large house with at least 5 bedrooms, but it must be low-maintenance. I prefer a quiet neighborhood but it needs to be in the heart of the city. My budget is $800K."
        
        # 处理初始查询
        state = await llm_service.process_user_input(initial_query)
        
        # 检查是否检测到矛盾
        assert state.get("has_ambiguities", False), "未检测到明显的矛盾"
        ambiguities = state.get("ambiguities", [])
        assert len(ambiguities) > 0, "未识别任何矛盾"
        
        print(f"\n检测到的矛盾: {json.dumps(ambiguities, indent=2)}")
        
        # 获取澄清问题
        if ambiguities and len(llm_service.chat_history) >= 2:
            clarification_question = llm_service.chat_history[-1]["content"]
            print(f"\n澄清问题: {clarification_question}")
            
            # 模拟用户对矛盾的回答
            user_response = "I guess location is more important to me than size. I could settle for a smaller place if it's in a central location. And maybe I could hire someone for maintenance if needed."
            print(f"\n用户回答: {user_response}")
            
            # 处理澄清回答
            final_state = await llm_service.process_user_input(user_response)
            
            # 检查偏好是否被正确更新
            preferences = final_state["userpreferences"]
            print(f"\n更新后的偏好: {json.dumps(preferences, indent=2)}")
            
            # 检查是否正确解决了矛盾（矛盾减少或消失）
            remaining_ambiguities = final_state.get("ambiguities", [])
            print(f"\n澄清后剩余矛盾: {len(remaining_ambiguities)}")
            assert len(remaining_ambiguities) < len(ambiguities), "澄清后矛盾未减少"

    @pytest.mark.asyncio
    async def test_comprehensive_ambiguity_detection(self, llm_service):
        """全面测试ambiguity_worker的检测能力，包含多种模糊和矛盾场景"""
        
        print("\n\n===== 开始测试：全面模糊与矛盾检测 =====")
        
        # 创建一系列包含不同类型模糊/矛盾的用户查询
        test_cases = [
            {
                "name": "位置模糊",
                "query": "I need a property in Sydney.",
                "expected_type": "vagueness",
                "expected_field": "location"
            },
            {
                "name": "价格模糊",
                "query": "I want an affordable apartment in Melbourne.",
                "expected_type": "vagueness",
                "expected_field": "price"
            },
            {
                "name": "位置矛盾",
                "query": "I want a quiet countryside property but within walking distance to the CBD.",
                "expected_type": "contradiction",
                "expected_field": "location"
            },
            {
                "name": "预算与期望矛盾",
                "query": "I'm looking for a luxury penthouse with high-end finishes. My budget is $500K.",
                "expected_type": "unrealistic",
                "expected_field": "price"
            },
            {
                "name": "规模与维护矛盾",
                "query": "I want a large house with 5+ bedrooms but it must be very low maintenance.",
                "expected_type": "contradiction",
                "expected_field": "size"
            }
        ]
        
        results = []
        
        # 执行每个测试用例
        for test_case in test_cases:
            print(f"\n\n测试: {test_case['name']}")
            print(f"用户查询: {test_case['query']}")
            
            # 清除之前的对话历史
            llm_service.chat_history = []
            
            # 处理查询
            state = await llm_service.process_user_input(test_case['query'])
            
            # 检查ambiguities
            ambiguities = state.get("ambiguities", [])
            print(f"检测到的模糊/矛盾: {json.dumps(ambiguities, indent=2)}")
            
            # 验证结果
            test_result = {
                "name": test_case["name"],
                "query": test_case["query"],
                "expected_type": test_case["expected_type"],
                "expected_field": test_case["expected_field"],
                "detected": False,
                "ambiguities": ambiguities
            }
            
            # 检查是否找到了预期的模糊/矛盾类型
            for ambiguity in ambiguities:
                if (ambiguity.get("type") == test_case["expected_type"] and 
                    ambiguity.get("field") == test_case["expected_field"]):
                    test_result["detected"] = True
                    test_result["clarification_question"] = ambiguity.get("clarification_question", "")
                    break
            
            results.append(test_result)
            
            if test_result["detected"]:
                print(f"✅ 成功检测到 {test_case['expected_type']} 类型的 {test_case['expected_field']} 模糊/矛盾")
                # 显示澄清问题
                if "clarification_question" in test_result:
                    print(f"澄清问题: {test_result['clarification_question']}")
            else:
                print(f"❌ 未能检测到 {test_case['expected_type']} 类型的 {test_case['expected_field']} 模糊/矛盾")
        
        # 总结结果
        success_count = sum(1 for r in results if r["detected"])
        print(f"\n\n测试结果总结: {success_count}/{len(test_cases)} 通过")
        for test_result in results:
            status = "✅ 通过" if test_result["detected"] else "❌ 失败"
            print(f"{status} {test_result['name']}")
        
        # 验证至少80%的测试通过
        assert success_count / len(test_cases) >= 0.8, f"模糊/矛盾检测成功率低于期望: {success_count}/{len(test_cases)}"

if __name__ == "__main__":
    import sys
    
    # 默认测试场景
    test_name = "test_clear_preferences_extraction"
    
    # 如果提供了命令行参数，使用指定的测试
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
    
    # 创建测试实例
    test_instance = TestPreferenceExtraction()
    
    # 获取测试方法
    test_method = getattr(test_instance, test_name)
    
    # 创建LLM服务
    llm_service = test_instance.llm_service()
    
    # 运行测试
    asyncio.run(test_method(llm_service))
    
    print(f"\n测试完成: {test_name}") 