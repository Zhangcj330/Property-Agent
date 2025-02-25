from openai import OpenAI
from typing import Dict, Tuple
import json
from .config import settings

class LLMService:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.GEMINI_API_KEY,
            base_url=settings.BASE_URL
        )

    async def chat_with_user(self, user_message: str, chat_history: list = None) -> str:
        """
        Generate a conversational response to user messages about real estate
        """
        try:
            # Construct the conversation context
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful real estate assistant. Help users find properties 
                    and answer their questions about real estate. Be friendly and concise. If users 
                    haven't provided enough information about their property requirements, ask for 
                    specific details about location, budget, or number of bedrooms."""
                }
            ]

            # Add chat history if provided
            if chat_history:
                messages.extend(chat_history[-5:])  # Keep last 5 messages for context

            # Add the current user message
            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=messages,
                temperature=0.7,  # More creative for conversation
                max_tokens=150    # Keep responses concise
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error in chat_with_user: {e}")
            return "I apologize, but I'm having trouble processing your request. Could you please try again?"

    async def extract_preferences(self, user_input: str) -> Dict:
        prompt = f"""
        Extract property preferences from this user input: "{user_input}"
        Return only a JSON object with these fields:
        - min_price (optional number)
        - max_price (number)
        - location (string, city or area)
        - min_bedrooms (number)
        - property_type (optional string)
        - must_have_features (list of strings)
        
        Format the response as valid JSON only, no other text.
        """

        try:
            response = self.client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[
                    {"role": "system", "content": "You are a helpful real estate assistant. Respond only with JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # Lower temperature for more consistent JSON output
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content.strip()
            
            # Clean the response to ensure it's valid JSON
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]  # Remove ```json and ``` markers
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]  # Remove ``` markers
            
            return json.loads(response_text)
            
        except Exception as e:
            print(f"Error in extract_preferences: {e}")
            # Return a default structure in case of error
            return {
                "max_price": 1000000,
                "location": "unknown",
                "min_bedrooms": 2,
                "must_have_features": []
            } 

    def _check_missing_information(self, preferences: Dict) -> list:
        """Check what important information is missing from preferences"""
        missing = []
        if not preferences.get('location'):
            missing.append('location')
        if not preferences.get('max_price'):
            missing.append('budget')
        if not preferences.get('min_bedrooms'):
            missing.append('number of bedrooms')
        return missing

    async def process_user_input(self, user_input: str, chat_history: list = None) -> Tuple[str, Dict]:
        """
        Process user input and return both a chat response and any extracted preferences
        """
        try:
            # First try to extract any property preferences
            preferences = await self.extract_preferences(user_input)
            print(preferences)
            # Check what information might be missing
            missing_info = self._check_missing_information(preferences)
            print(missing_info)
            # Generate appropriate response
            if missing_info:
                # Ask for missing information
                response = f"I'd like to help you find the perfect property. Could you please tell me about your preferred {' and '.join(missing_info)}?"
            else:
                # Generate response based on complete preferences
                response = await self.chat_with_user(
                    f"Generate a response for a user looking for a {preferences.get('min_bedrooms')}-bedroom property "
                    f"in {preferences.get('location')} with a budget of ${preferences.get('max_price'):,}",
                    chat_history
                )
            print(response)
            return response, preferences

        except Exception as e:
            print(f"Error in process_user_input: {e}")
            return "I'm here to help you find properties. What kind of property are you looking for?", {} 