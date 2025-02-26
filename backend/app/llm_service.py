from typing import Dict, Tuple, List, Any, Annotated
from pydantic import BaseModel, Field
import operator
import json
from typing_extensions import TypedDict
from enum import Enum

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, MessagesState, START, END
from .config import settings
from .models import UserPreferences
from .services.property_scraper import PropertyScraper

# Graph state
class State(TypedDict):
    messages: MessagesState
    preferences: Dict
    current_field: str
    completed_fields: List[str]
    is_complete: bool
    chat_history: List[Dict]  # Add chat history to state

# Worker state
class WorkerState(TypedDict):
    field: str  # Changed from section to field
    completed_fields: Annotated[list, operator.add]

# Default LLM
llm = ChatOpenAI(
    api_key=settings.GEMINI_API_KEY,
    base_url=settings.BASE_URL,
    model="gemini-2.0-flash"
)

# Structured output parser
def create_agent_graph():
    # Worker: Extract preferences from input
    def extract_worker(state: State) -> State:
        """Extract preferences from user input using conversation context"""
        parser = JsonOutputParser(pydantic_object=UserPreferences)
        messages = state["messages"]
        current_preferences = state["preferences"]
        chat_history = state.get("chat_history", [])
        
        if not messages or not isinstance(messages[-1], HumanMessage):
            return state
            
        try:
            # First, extract preferences from the entire conversation context
            extraction_prompt = """You are an expert Australian real estate consultant and demographic analyst.
            Analyze the conversation to:
            1. Extract real estate preferences
            2. Infer demographic details (age, income, lifestyle, family status) from the conversation
            3. Recommend suitable suburbs based on the demographic profile and preferences
            
            For location recommendations:
            - Consider factors like proximity to schools, public transport, entertainment
            - Match suburbs to the likely demographic profile
            - Format location as STATE-SUBURB-POSTCODE (e.g., "NSW-Chatswood-2067")
            
            You must respond with a JSON object containing:
            {
                "location": "STATE-SUBURB-POSTCODE",
                "suburb": "suburb name",
                "state": "state code",
                "postcode": number,
                "max_price": number or null,
                "min_price": number or null,
                "min_bedrooms": number or null,
                "property_type": "house/apartment/etc" or null,
                "must_have_features": [],
                "demographic_analysis": {
                    "likely_age_group": "string",
                    "estimated_income": "string",
                    "lifestyle_preferences": "string",
                    "family_status": "string"
                },
                "suburb_recommendation_reason": "string"
            }
            
            If the user hasn't specified a location but has given enough context, recommend a suitable suburb.
            Include only fields that are clearly mentioned or can be confidently inferred.
            DO NOT generate questions or conversational responses."""
            
            # Build conversation context
            conversation = "\nConversation:\n"
            for msg in chat_history:
                conversation += f"{msg['role'].title()}: {msg['content']}\n"
            
            # Add current message
            conversation += f"User: {messages[-1].content}\n"
            
            # Extract preferences
            response = llm.invoke([
                SystemMessage(content=extraction_prompt),
                HumanMessage(content=conversation)
            ])
            
            print("Conversation context:", conversation)
            print("LLM response:", response.content.strip())
            
            # Parse extracted preferences
            new_prefs = parser.invoke(response)
            print("Parsed preferences:", new_prefs)
            
            # Update preferences, keeping existing values if not in new extraction
            merged_preferences = {**current_preferences}
            for key, value in new_prefs.items():
                if value is not None:  # Only update if new value is not None
                    merged_preferences[key] = value
            
            state["preferences"] = merged_preferences
            
        except Exception as e:
            print(f"Error in extract_worker: {e}")
        
        return state

    # Worker: Check if all required fields are present
    def check_worker(state: State) -> State:
        """Check if all required preferences are present"""
        preferences = state["preferences"]
        required_fields = ["location", "max_price", "property_type"]
        
        # Check each required field
        missing = []
        for field in required_fields:
            if not preferences.get(field):
                missing.append(field)
        
        if missing:
            # Set the first missing field as current field
            state["current_field"] = missing[0]
            state["is_complete"] = False
            print(f"Missing fields: {missing}")
        else:
            state["is_complete"] = True
            print("All required fields collected:", preferences)
            
        return state

    # Worker: Generate follow-up question
    def question_worker(state: State) -> State:
        """Generate follow-up question for missing information"""
        current_field = state["current_field"]
        preferences = state["preferences"]
        
        # Question templates based on UserPreferences fields
        questions = {
            "location": "What city or area are you interested in?",
            "max_price": "What's your maximum budget for the property?",
            "min_beds": "How many bedrooms do you need?",
            "property_type": "What type of property are you looking for?",
            "must_have_features": "Any specific features you're looking for?"
        }
        
        # Generate natural follow-up question
        response = llm.invoke([
            SystemMessage(content="""You are a helpful real estate assistant.
            Ask for the missing information naturally and conversationally.
            Keep the question focused and clear."""),
            HumanMessage(content=f"""
            The user's current preferences: {json.dumps(preferences)}
            Ask about: {questions[current_field]}
            """)
        ])
        
        # Add question to messages
        state["messages"].append(AIMessage(content=response.content))
        return state

    # Create the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("extract", extract_worker)
    workflow.add_node("check", check_worker)
    workflow.add_node("question", question_worker)
    
    # Add edges
    workflow.add_edge(START, "extract")
    workflow.add_edge("extract", "check")
    workflow.add_conditional_edges(
        "check",
        lambda x: END if x["is_complete"] else "question"
    )
    workflow.add_edge("question", END)
    
    return workflow.compile()

class LLMService:
    def __init__(self):
        self.graph = create_agent_graph()
        self.chat_history = []  # Initialize chat history
        self.property_scraper = PropertyScraper()  # Initialize property scraper
    
    async def process_user_input(self, user_input: str, preferences: Dict = None) -> Tuple[str, Dict]:
        """Process user input through the graph"""
        
        # Add user message to chat history
        self.chat_history.append({"role": "user", "content": user_input})
        
        # Initialize state with chat history
        state = State(
            messages=[HumanMessage(content=user_input)],
            preferences=preferences or {},
            current_field=None,
            completed_fields=[],
            is_complete=False,
            chat_history=self.chat_history
        )
        
        # Run the graph
        final_state = self.graph.invoke(state)
        print("Final state:", final_state)
        
        # Get last response and updated preferences
        last_message = final_state["messages"][-1].content if final_state["messages"] else ""
        updated_preferences = final_state["preferences"]
        
        # If we have complete preferences, search for properties
        if final_state["is_complete"]:
            try:    
                # Search for properties using scraper
                properties = await self.property_scraper.search_properties(
                    location=updated_preferences.get("location"),
                    min_price=updated_preferences.get("min_price"),
                    max_price=updated_preferences.get("max_price"),
                    min_beds=updated_preferences.get("min_bedrooms"),
                    property_type=updated_preferences.get("property_type"),
                    max_results=5  # Limit results
                )
                
                # Format property results into response
                if properties:
                    property_summary = "\n\nHere are some properties that match your criteria:\n"
                    for i, prop in enumerate(properties, 1):
                        property_summary += f"\n{i}. {prop['address']}"
                        if prop['price']:
                            property_summary += f"\nPrice: {prop['price']}"
                        property_summary += f"\nBedrooms: {prop['bedrooms']}"
                        if prop['property_type']:
                            property_summary += f"\nType: {prop['property_type']}"
                        property_summary += "\n"
                    
                    last_message += property_summary
                else:
                    last_message += "\n\nI couldn't find any properties matching your criteria at the moment."
                
            except Exception as e:
                print(f"Error searching properties: {e}")
                last_message += "\n\nI encountered an error while searching for properties."
        
        # Add assistant response to chat history
        self.chat_history.append({"role": "assistant", "content": last_message})
        
        return last_message, updated_preferences