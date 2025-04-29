from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.models import ChatSession, ChatMessage
from app.services.chat_storage import ChatStorageService

class PreferenceService:
    """Service class for extracting and managing user preferences"""
    
    def __init__(self):
        self.chat_storage = ChatStorageService()
        self.llm = ChatGoogleGenerativeAI(
            api_key=settings.GEMINI_API_KEY,
            base_url=settings.BASE_URL,
            model="gemini-2.0-flash",
            temperature=0.1,
            max_output_tokens=2048
        )
    
    async def process_user_input(
        self, 
        session_id: str, 
        user_message: str
    ) -> Tuple[Dict, Dict, Optional[Dict]]:
        """Process user input to extract and update preferences and search parameters
        
        Returns:
            Tuple containing:
            - preferences: Dict of updated preferences
            - search_params: Dict of updated search parameters
        """
        
        # Get session history
        session = await self.chat_storage.get_session(session_id)
        if not session:
            session = await self.chat_storage.create_session(session_id)
        
        # Get current state
        current_preferences = session.preferences or {}
        current_search_params = session.search_params or {}
        
        # Prepare context from conversation history
        context = self._prepare_conversation_context(session.messages)
        
        # Extract and update preferences and search parameters
        preferences, search_params = await self._process_with_llm(
            context=context,
            user_message=user_message,
            current_preferences=current_preferences,
            current_search_params=current_search_params
        )
        
        # Save updates to session
        await self.chat_storage.update_session_state(
            session_id=session_id,
            preferences=preferences,
            search_params=search_params,
        )
        
        # Log updates in chat history
        await self._log_updates(
            session_id=session_id,
            old_preferences=current_preferences,
            new_preferences=preferences,
            old_search_params=current_search_params,
            new_search_params=search_params,
        )
        
        return preferences, search_params
    
    def _prepare_conversation_context(self, messages: List[ChatMessage]) -> str:
        """Prepare conversation context from recent messages"""
        if not messages:
            return "No dialogue history"
        
        # Get last 5 rounds of user-assistant dialogue
        user_assistant_msgs = [msg for msg in messages if msg.role in ["user", "assistant"]]
        total_rounds = len(user_assistant_msgs) // 2
        rounds_to_include = min(total_rounds, 5)
        
        start_idx = max(0, len(user_assistant_msgs) - (rounds_to_include * 2))
        recent_dialog = user_assistant_msgs[start_idx:]
        
        # Format context
        context = "Recent Dialogue:\n"
        current_round = total_rounds - rounds_to_include + 1
        
        for i in range(0, len(recent_dialog), 2):
            context += f"--- Round {current_round} ---\n"
            context += f"User: {recent_dialog[i].content}\n"
            if i + 1 < len(recent_dialog):
                context += f"Assistant: {recent_dialog[i+1].content}\n"
            context += "\n"
            current_round += 1
        
        return context
    
    async def _process_with_llm(
        self, 
        context: str,
        user_message: str,
        current_preferences: Dict,
        current_search_params: Dict
    ) -> Tuple[Dict, Dict, Optional[Dict]]:
        """Use LLM to process user input and generate updated preferences and search parameters"""
        
        # Format current state
        current_state = self._format_current_state(current_preferences, current_search_params)
        
        prompt = f"""You are an expert property preference analyst. Your task is to analyze the conversation history and user's latest message to understand their property requirements and preferences.

Current State:
{current_state}

Conversation Context:
{context}

User's Latest Message:
{user_message}

Task:
Analyze the information and generate a structured representation of the user's current preferences and search parameters. "preferences" refer to qualitative desires and "search parameters" refer to quantifiable criteria. Consider both explicit statements and implicit preferences that can be reasonably inferred from the context.

Key Analysis Points:
1. Consider the evolution of preferences across the conversation
2. Look for changes or refinements in requirements
3. Pay attention to both positive ("I want", "I like") and negative ("I don't want", "too noisy") expressions
4. Consider the strength of preferences when setting importance values
5. Only include qualitative preferences and search parameters quantifiable parameters that have *clear supporting evidence*
6. Preserve existing values unless there's clear indication for change
7. Detect and handle ambiguous statements that needs clarification, for example:
    - Broad location names (e.g., "Sydney", "North Shore")
    - Vague price ranges (e.g., "affordable", "expensive")

Data Structure Requirements:

1. Preferences must follow this exact structure:
{{
  "preferences": {{
    "Style": {{ "preference": "modern", "importance": 0.8 }},
    "Environment": {{ "preference": "quiet suburb", "importance": 0.7 }},
    ...
  }}
}}

Valid preference categories:
- Style (architectural and design preferences)
- Environment (area characteristics, noise, greenery)
- Features (specific property features)
- Quality (construction and maintenance)
- Layout (floor plan and space arrangement)
- Transport (accessibility and commute)
- Location (specific area preferences)
- Schools (education facilities)
- Community (neighborhood characteristics)
- Investment (potential returns and growth)

2. Search parameters must follow this exact structure:
{{
  "search_params": {{
    "location": ["NSW-CHATSWOOD-2067", "NSW-MILLSONS-POINT-2061"], // List suburbs with state-suburb-postcode format
    "min_price": 1000000,                 // User's minimum budget, numeric only (no commas)
    "max_price": 1500000,                 // User's maximum budget, numeric only (no commas)
    "min_bedrooms": 3,                    // Minimum number of bedrooms
    "min_bathrooms": 2,                   // Minimum number of bathrooms
    "property_type": ["house", "apartment"], // Desired property types (e.g., house, apartment, townhouse, duplex)
    "car_parks": 1,                       // Required number of car parks
    "land_size_from": 300,                // Minimum land size (sqm)
    "land_size_to": 600                   // Maximum land size (sqm)
  }}
}}

Important Rules:
1. ONLY include preferences and parameters that have clear evidence from the conversation
2. Do NOT generate placeholder or assumed values
3. For location format, strictly follow: STATE-SUBURB-POSTCODE
5. Importance values must be between 0.1 (low) and 1.0 (high)
6. Property types must be lowercase: house, apartment, unit, townhouse, villa, rural
7. All numeric values must be appropriate for their context

Generate a valid JSON response with the above structure. Include ONLY fields that have clear supporting evidence from the conversation."""
        
        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Based on the conversation history and current state shown above, generate the structured representation of the user's current preferences and search parameters. Include ONLY well-supported preferences and parameters, and identify any ambiguous information that needs clarification.")
        ])
        print("LLM Response:")
        print(response.content)
        try:
            # Extract and parse JSON
            content = self._extract_json_from_response(response.content)
            result = json.loads(content)
            
            # Extract preferences, search parameters, and ambiguity information
            preferences = result.get("preferences", {})
            search_params = result.get("search_params", {})
            
            # Normalize and validate values
            search_params = self._normalize_search_params(search_params)
            
            return preferences, search_params
        
            
        except Exception as e:
            print(f"Error processing LLM response: {e}")
            return current_preferences, current_search_params, None
    
    def _format_current_state(self, preferences: Dict, search_params: Dict) -> str:
        """Format current preferences and search parameters for LLM prompt"""
        state = "Current Preferences:\n"
        
        for category, pref in preferences.items():
            if isinstance(pref, dict) and "preference" in pref:
                importance = pref.get("importance", 0.5)
                state += f"- {category}: {pref['preference']} (Importance: {importance})\n"
        
        state += "\nCurrent Search Parameters:\n"
        for param, value in search_params.items():
            if value is not None:
                state += f"- {param}: {value}\n"
        
        return state
    
    def _extract_json_from_response(self, content: str) -> str:
        """Extract JSON from LLM response"""
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Clean up common formatting issues
        content = content.replace("'", '"')
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        
        return content
    
    def _normalize_search_params(self, params: Dict) -> Dict:
        """Normalize and validate search parameters"""
        normalized = {}
        
        for param, value in params.items():
            if param == "location":
                # Ensure location is a list and properly formatted
                if isinstance(value, str):
                    value = [value]
                if isinstance(value, list):
                    normalized[param] = [
                        loc.upper() for loc in value 
                        if isinstance(loc, str) and len(loc.split('-')) == 3
                    ]
            elif param == "property_type":
                # Ensure property_type is a list
                if isinstance(value, str):
                    value = [value]
                if isinstance(value, list):
                    normalized[param] = [str(t).lower() for t in value]
            elif param in ["min_price", "max_price", "land_size_from", "land_size_to"]:
                # Ensure numeric values
                try:
                    normalized[param] = float(value)
                except (TypeError, ValueError):
                    continue
            elif param in ["min_bedrooms", "min_bathrooms", "car_parks"]:
                # Ensure integer values
                try:
                    normalized[param] = int(value)
                except (TypeError, ValueError):
                    continue

        
        return normalized
    
    async def _log_updates(
        self,
        session_id: str,
        old_preferences: Dict,
        new_preferences: Dict,
        old_search_params: Dict,
        new_search_params: Dict
    ) -> None:
        """Log preference and search parameter updates to chat history"""
        updates = []
        
        # Check for preference changes
        for category, new_pref in new_preferences.items():
            old_pref = old_preferences.get(category, {})
            if new_pref != old_pref:
                updates.append(f"• Updated {category}: {new_pref.get('preference')} "
                             f"(Importance: {new_pref.get('importance', 0.5):.1f})")
        
        # Check for search parameter changes
        for param, new_value in new_search_params.items():
            old_value = old_search_params.get(param)
            if new_value != old_value:
                updates.append(f"• Updated {param}: {new_value}")
        
        if updates :
            log_message = []
            if updates:
                log_message.append("I've updated your preferences and search parameters:")
                log_message.extend(updates)
            
            await self.chat_storage.save_message(
                session_id,
                ChatMessage(
                    role="assistant",
                    content="\n".join(log_message),
                    timestamp=datetime.now()
                )
            )

# Helper functions for external use
async def process_user_preferences(
    session_id: str,
    user_message: str
) -> Dict:
    """Process user message to extract and update preferences and search parameters
    
    Args:
        session_id: Session ID
        user_message: User message content
        
    Returns:
        Dictionary containing updated preferences, search parameters, and any ambiguity information
    """
    service = PreferenceService()
    try:
        preferences, search_params = await service.process_user_input(session_id, user_message)
        return {
            "preferences": preferences,
            "search_params": search_params,
        }
    except Exception as e:
        print(f"Error processing user preferences: {e}")
        return {
            "preferences": {},
            "search_params": {},
        }
