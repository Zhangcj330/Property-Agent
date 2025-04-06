from typing import Dict, List, Optional, Any, Union, Tuple
from pydantic import BaseModel
from datetime import datetime
import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.config import settings
from app.models import UserPreferences, UserPreference, ChatSession, ChatMessage, PropertySearchRequest
from app.services.chat_storage import ChatStorageService

class PreferenceUpdate(BaseModel):
    """Model for updating preferences"""
    preference_type: str  # "explicit" or "implicit"
    category: str  # Style, Environment, Features, etc.
    value: str
    importance: float = 0.5
    reason: Optional[str] = None
    source_message: Optional[str] = None

class SearchParamUpdate(BaseModel):
    """Model for updating search parameters"""
    param_name: str  # location, min_price, max_price, etc.
    value: Any
    reason: Optional[str] = None

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
    
    async def extract_from_context(
        self, 
        session_id: str, 
        recent_message: str
    ) -> Tuple[List[PreferenceUpdate], List[SearchParamUpdate]]:
        """Extract user preferences and search parameters from recent message and conversation context"""
        
        # Get session history
        session = await self.chat_storage.get_session(session_id)
        if not session:
            return [], []
        
        # Get current preferences and search parameters
        current_preferences = session.preferences or {}
        current_search_params = session.search_params or {}
        
        # Prepare context (last 5 rounds of dialogue)
        context = self._prepare_conversation_context(session.messages)
        
        # Use LLM to extract preferences and search parameters
        preferences, search_params = await self._extract_with_llm(
            context, 
            recent_message, 
            current_preferences, 
            current_search_params
        )
        
        return preferences, search_params
    
    async def update_user_preferences(
        self, 
        session_id: str, 
        preference_updates: List[PreferenceUpdate]
    ) -> UserPreferences:
        """Update user preferences and save to storage"""
        
        # Get current preferences
        session = await self.chat_storage.get_session(session_id)
        if not session:
            session = await self.chat_storage.create_session(session_id)
        
        current_preferences = session.preferences or {}
        
        # Apply updates
        updated_preferences = self._apply_preference_updates(current_preferences, preference_updates)
        
        # Save updated preferences
        await self.chat_storage.update_session_state(
            session_id=session_id,
            preferences=updated_preferences,
            search_params=session.search_params
        )
        
        # Integrate batch record preference updates
        if preference_updates:
            log_messages = []
            for update in preference_updates:
                msg = f"• {update.category}: {update.value} ({update.preference_type})"
                if update.reason:
                    msg += f" - {update.reason}"
                log_messages.append(msg)
            
            combined_log = f"Preferences Updated:\n" + "\n".join(log_messages)
            
            await self.chat_storage.save_message(
                session_id, 
                ChatMessage(
                    role="system",
                    content=combined_log,
                    timestamp=datetime.now()
                )
            )
        
        return updated_preferences
    
    async def update_search_params(
        self, 
        session_id: str, 
        search_param_updates: List[SearchParamUpdate]
    ) -> Dict:
        """Update search parameters and save to storage"""
        
        # Get current search parameters
        session = await self.chat_storage.get_session(session_id)
        if not session:
            session = await self.chat_storage.create_session(session_id)
        
        current_search_params = session.search_params or {}
        
        # Apply updates
        updated_search_params = self._apply_search_param_updates(current_search_params, search_param_updates)
        
        # Save updated search parameters
        await self.chat_storage.update_session_state(
            session_id=session_id,
            preferences=session.preferences,
            search_params=updated_search_params
        )
        
        # Integrate batch record search parameter updates
        if search_param_updates:
            log_messages = []
            for update in search_param_updates:
                msg = f"• {update.param_name}: {update.value}"
                if update.reason:
                    msg += f" - {update.reason}"
                log_messages.append(msg)
            
            combined_log = f"Search Parameters Updated:\n" + "\n".join(log_messages)
            
            await self.chat_storage.save_message(
                session_id, 
                ChatMessage(
                    role="system",
                    content=combined_log,
                    timestamp=datetime.now()
                )
            )
        
        return updated_search_params
    
    def _prepare_conversation_context(self, messages: List[ChatMessage]) -> str:
        """Prepare conversation context - ensure getting last 5 rounds of dialogue (each including user and assistant messages)"""
        if not messages:
            return "No dialogue history"
        
        # Filter out system messages, only keep user and assistant messages
        user_assistant_msgs = [msg for msg in messages if msg.role in ["user", "assistant"]]
        
        # Calculate rounds and get last 5 rounds of dialogue
        total_rounds = len(user_assistant_msgs) // 2
        rounds_to_include = min(total_rounds, 5)
        
        # Get recent N rounds of messages (each round includes 2 messages, user+assistant)
        start_idx = max(0, len(user_assistant_msgs) - (rounds_to_include * 2))
        recent_dialog = user_assistant_msgs[start_idx:]
        
        # Format context
        context = "Recent Dialogue:\n"
        current_round = total_rounds - rounds_to_include + 1
        
        for i in range(0, len(recent_dialog), 2):
            # Ensure we have a pair of messages (user+assistant)
            if i + 1 < len(recent_dialog):
                context += f"--- Round {current_round} ---\n"
                context += f"User: {recent_dialog[i].content}\n"
                context += f"Assistant: {recent_dialog[i+1].content}\n\n"
                current_round += 1
            else:
                # Last may be only user message without assistant reply
                context += f"--- Round {current_round} ---\n"
                context += f"User: {recent_dialog[i].content}\n"
        
        return context
    
    async def _extract_with_llm(
        self, 
        context: str,
        recent_message: str,
        current_preferences: Dict,
        current_search_params: Dict
    ) -> Tuple[List[PreferenceUpdate], List[SearchParamUpdate]]:
        """Use LLM to extract user preferences and search parameters"""
        
        # Convert current preferences to readable format
        current_prefs_str = ""
        if current_preferences:
            current_prefs_str = "Current Known Preferences:\n"
            for category, pref in current_preferences.items():
                if pref and "preference" in pref and pref["preference"]:
                    importance = pref.get("importance", 0.5)
                    importance_text = "High" if importance > 0.7 else ("Medium" if importance > 0.3 else "Low")
                    current_prefs_str += f"- {category}: {pref['preference']} (Importance: {importance_text})\n"
        
        # Convert current search parameters to readable format
        current_params_str = ""
        if current_search_params:
            current_params_str = "Current Search Parameters:\n"
            for param, value in current_search_params.items():
                if value is not None:
                    current_params_str += f"- {param}: {value}\n"
        
        prompt = f"""You are a professional property preference and search requirement analyst. Your task is to analyze conversations and extract both explicit and implicit preferences, as well as specific search parameters.

Key Analysis Areas:

1. User Preferences:
   A. Explicit Preferences - Directly stated requirements:
      - Clear statements: "I want...", "I need...", "I'm looking for..."
      - Direct preferences: "I like modern style", "I prefer quiet areas"
      
   B. Implicit Preferences - Inferred from context:
      - Negative reactions: "too noisy" → preference for quiet
      - Comparative statements: "better if closer to station" → preference for good transport
      - Questions about features: "Does it have a garden?" → potential interest in outdoor space

2. Search Parameters:
   A. Location Parameters:
      Format: ["NSW-chatswood-2067", "NSW-epping-2121"]
      
      Special Handling for Ambiguous Locations:
      - Large Areas (e.g., "Sydney", "North Shore", "Eastern Suburbs")
      - Return Format for Ambiguous Locations:
        {{
          "param_name": "location",
          "value": {{
            "term": "ambiguous_term",
            "suggestions": ["area1", "area2", "area3"],
            "context": "area characteristics explanation"
          }}
        }}
      
      Example Suggestions:
      - "North Shore": Chatswood (Asian community), Lane Cove (family-friendly)
      - "Eastern Suburbs": Bondi (beach lifestyle), Double Bay (luxury living)

   B. Other Parameters:
      - Price Range: min_price, max_price (e.g., 1500000, 2000000)
      - Bedrooms: min_bedrooms (e.g., 3)
      - Bathrooms: min_bathrooms (e.g., 2)
      - Property Type: property_type (e.g., ["house", "apartment"])
      - Parking: car_parks (e.g., 2)
      - Land Size: land_size_from, land_size_to (e.g., 300, 800)

3. Importance Level Analysis:
   A. High Importance (0.7-1.0):
      - Must-have statements: "must have", "need", "has to be"
      - Deal-breakers: "won't consider without", "absolutely need"
      - Strong emphasis: "very important", "essential"
   
   B. Medium Importance (0.4-0.6):
      - Preferences: "would like", "prefer", "want"
      - General desires: "looking for", "interested in"
      - Regular statements without strong emphasis
   
   C. Low Importance (0.1-0.3):
      - Optional features: "would be nice", "if possible"
      - Casual mentions: "maybe", "could have"
      - Tentative interest: "might be good"

4. Preference Categories:
   - Style: Architectural and design preferences
   - Environment: Surrounding area characteristics
   - Features: Property amenities and facilities
   - Quality: Construction and maintenance standards
   - Layout: Space arrangement and floor plan
   - Transport: Accessibility and commute options
   - Location: Area and position preferences
   - Schools: Educational facility access
   - Community: Neighborhood characteristics
   - Investment: Investment potential and return on investment

Current Known Information:
{current_prefs_str}
{current_params_str}

Conversation Context:
{context}

User's Recent Message:
{recent_message}

Provide ONLY a valid JSON response in this format:
{{
  "preferences": [
    {{
      "preference_type": "explicit|implicit",
      "category": "Style|Environment|Features|...",
      "value": "specific preference content",
      "importance": 0.1-1.0,
      "reason": "detailed extraction/inference reason"
    }}
  ],
  "search_params": [
    {{
      "param_name": "location|min_price|max_price|...",
      "value": "parameter value",
      "reason": "parameter extraction reason"
    }}
  ]
}}

Example Response:
{{
  "preferences": [
    {{
      "preference_type": "explicit",
      "category": "Style",
      "value": "modern",
      "importance": 0.6,
      "reason": "User directly expressed preference for modern style"
    }},
    {{
      "preference_type": "implicit",
      "category": "Transport",
      "value": "near_station",
      "importance": 0.4,
      "reason": "User asked about distance to train station"
    }}
  ],
  "search_params": [
    {{
      "param_name": "location",
      "value": "NSW-chatswood-2067",
      "reason": "User specifically mentioned Chatswood area"
    }},
    {{
      "param_name": "min_bedrooms",
      "value": 3,
      "reason": "User requested three bedrooms"
    }}
  ]
}}

If no information is detected for a category, return empty array []."""
        
        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Context: {context}\nUser's Recent Message: {recent_message}")
        ])
        
        try:
            # Try to parse entire response directly first
            try:
                content = response.content
                # Try to extract JSON from code block if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try to find any JSON-like structure
                json_match = re.search(r'{\s*"preferences":\s*\[.*?\],\s*"search_params":\s*\[.*?\]\s*}', response.content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                else:
                    print("Could not find valid JSON in response")
                    return [], []
            
            # Parse results
            preferences = [PreferenceUpdate(**item) for item in result.get("preferences", [])]
            search_params = [SearchParamUpdate(**item) for item in result.get("search_params", [])]
            
            return preferences, search_params
            
        except Exception as e:
            print(f"Error parsing update: {e}")
            print(f"Raw response: {response.content}")
            return [], []
    
    def _apply_preference_updates(
        self, 
        current_preferences: Dict, 
        updates: List[PreferenceUpdate]
    ) -> Dict:
        """Apply preference updates to current preferences"""
        updated_preferences = dict(current_preferences)
        
        for update in updates:
            category = update.category
            
            # Check if category exists
            if category not in updated_preferences:
                updated_preferences[category] = {}
            # Update preference with only necessary fields
            updated_preferences[category] = {
                "preference": update.value,
                "importance": update.importance
            }
            
        return updated_preferences
    
    def _apply_search_param_updates(
        self, 
        current_params: Dict, 
        updates: List[SearchParamUpdate]
    ) -> Dict:
        """Apply search parameter updates to current parameters"""
        updated_params = dict(current_params)
        
        for update in updates:
            param_name = update.param_name
            
            # Handle location ambiguity case
            if param_name == "location_ambiguity":
                updated_params["location_ambiguity"] = update.value
                continue
                
            # Special handling for some parameters that need array format
            if param_name in ["location", "property_type"]:
                # If it's a string, convert to single-element array
                if isinstance(update.value, str):
                    value = [update.value]
                elif isinstance(update.value, list):
                    value = update.value
                else:
                    continue  # Skip invalid values
                
                # Keep existing value's uniqueness
                if param_name in updated_params and updated_params[param_name]:
                    existing_values = updated_params[param_name]
                    # Merge and remove duplicates
                    updated_params[param_name] = list(set(existing_values + value))
                else:
                    updated_params[param_name] = value
            else:
                # For other parameters, directly update
                updated_params[param_name] = update.value
        
        return updated_params

# Tool functions for Agent use
async def extract_preferences_and_search_params(
    session_id: str,
    user_message: str,
    service: PreferenceService = None
) -> Dict:
    """Extract preferences and search parameters from user message
    
    Args:
        session_id: Session ID
        user_message: User message content
        service: Optional PreferenceService instance. If not provided, a new one will be created.
        
    Returns:
        Dictionary containing extracted preferences and search parameters
    """
    # Use provided service or create a new one
    if service is None:
        service = PreferenceService()
    
    try:
        preferences, search_params = await service.extract_from_context(session_id, user_message)
        
        # Update preferences and search parameters
        updated_preferences = None
        if preferences:
            updated_preferences = await service.update_user_preferences(session_id, preferences)
        
        updated_search_params = None
        if search_params:
            updated_search_params = await service.update_search_params(session_id, search_params)
        
        return {
            "preferences": [pref.dict() for pref in preferences],
            "search_params": [param.dict() for param in search_params],
            "updated_preferences": updated_preferences,
            "updated_search_params": updated_search_params
        }
    except Exception as e:
        print(f"Error in extract_preferences_and_search_params: {str(e)}")
        return {
            "preferences": [],
            "search_params": [],
            "updated_preferences": None,
            "updated_search_params": None
        }

async def get_current_preferences_and_search_params(
    session_id: str
) -> Dict:
    """Get current preferences and search parameters for a session
    
    Args:
        session_id: Session ID
        
    Returns:
        Dictionary containing current preferences and search parameters
    """
    service = PreferenceService()
    session = await service.chat_storage.get_session(session_id)
    if not session:
        return {"preferences": {}, "search_params": {}}
    
    return {
        "preferences": session.preferences or {},
        "search_params": session.search_params or {}
    }

async def infer_preference_from_rejection(
    session_id: str,
    rejection_message: str,
    property_details: Dict
) -> Dict:
    """Infer preferences from rejection message
    
    Args:
        session_id: Session ID
        rejection_message: User's rejection message, e.g., "This recommendation is not suitable"
        property_details: Details of the rejected property
        
    Returns:
        Dictionary containing inferred preferences and updated preferences
    """
    service = PreferenceService()
    
    # Get or create session
    session = await service.chat_storage.get_session(session_id)
    if not session:
        session = await service.chat_storage.create_session(session_id)
        # Add initial message to provide context
        await service.chat_storage.save_message(
            session_id,
            ChatMessage(
                role="user",
                content=rejection_message,
                timestamp=datetime.now()
            )
        )
    
    # Prepare context
    context = service._prepare_conversation_context(session.messages)
    
    # Add rejected property information
    property_context = "Rejected Property Information:\n"
    for key, value in property_details.items():
        property_context += f"- {key}: {value}\n"
    
    # Rejection-specific prompt
    prompt = f"""You are a property preference analyst. Convert this rejection message into preferences:

Input: "This house is too noisy and old-fashioned"
Output: [
  {{
    "preference_type": "implicit",
    "category": "Environment",
    "value": "quiet_area",
    "importance": 0.8,
    "reason": "User rejected property for being too noisy",
    "source_message": "This house is too noisy and old-fashioned"
  }},
  {{
    "preference_type": "implicit",
    "category": "Style",
    "value": "modern",
    "importance": 0.7,
    "reason": "User rejected old-fashioned style",
    "source_message": "This house is too noisy and old-fashioned"
  }}
]

Rules:
1. Convert each complaint into a positive preference
2. Use high importance (0.7-1.0) for strong rejections
3. Include all required fields
4. Return only the JSON array

Now convert this rejection:
"{rejection_message}"

Property details:
{property_details}"""

    response = service.llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Rejection Message: {rejection_message}\nProperty Information: {property_context}")
    ])
    
    try:
        content = response.content
        print(f"Raw LLM response: {content}")  # Debug output
        
        # Try to extract JSON from code block if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Clean up common formatting issues
        content = content.replace("'", '"')  # Replace single quotes with double quotes
        content = re.sub(r',\s*]', ']', content)  # Remove trailing commas
        content = re.sub(r',\s*}', '}', content)  # Remove trailing commas in objects
        
        # Try to parse the cleaned content
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e1:
            print(f"Initial JSON parse failed: {e1}")  # Debug output
            
            # Try to find any JSON array structure
            json_match = re.search(r'\[\s*{.*?}\s*\]', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as e2:
                    print(f"JSON array extraction failed: {e2}")  # Debug output
                    
                    # Last resort: try to fix common JSON formatting issues
                    try:
                        # Replace newlines and extra spaces
                        json_str = re.sub(r'\s+', ' ', json_str)
                        # Ensure property names are quoted
                        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                        result = json.loads(json_str)
                    except json.JSONDecodeError as e3:
                        print(f"JSON cleanup failed: {e3}")  # Debug output
                        return {"preferences": [], "updated_preferences": {}}
            else:
                print("Could not find valid JSON array in response")  # Debug output
                return {"preferences": [], "updated_preferences": {}}
        
        # Validate result structure
        if not isinstance(result, list):
            print(f"Invalid result structure. Expected list, got: {type(result)}")  # Debug output
            return {"preferences": [], "updated_preferences": {}}
        
        # Ensure all required fields are present
        validated_results = []
        for item in result:
            if all(key in item for key in ["preference_type", "category", "value", "importance", "reason", "source_message"]):
                validated_results.append(item)
            else:
                print(f"Skipping invalid preference item: {item}")  # Debug output
        
        if not validated_results:
            print("No valid preferences found after validation")  # Debug output
            return {"preferences": [], "updated_preferences": {}}
        
        # Apply these implicit preferences
        updates = [PreferenceUpdate(**item) for item in validated_results]
        updated_preferences = {}
        
        if updates:
            updated_preferences = await service.update_user_preferences(session_id, updates)
        
        return {
            "preferences": [update.dict() for update in updates],
            "updated_preferences": updated_preferences
        }
    except Exception as e:
        print(f"Error inferring preferences from rejection: {str(e)}")  # Enhanced error output
        print(f"Response content type: {type(response.content)}")  # Debug output
        return {"preferences": [], "updated_preferences": {}}
