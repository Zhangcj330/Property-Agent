from typing import Dict, Tuple, List, Annotated, Optional
from pydantic import BaseModel
import operator
import json
from typing_extensions import TypedDict
from enum import Enum

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, MessagesState, START, END
from app.config import settings
from app.services.property_scraper import PropertyScraper
from app.models import UserPreferences, PropertySearchRequest

# Graph state
class State(TypedDict):
    messages: MessagesState
    userpreferences: Dict
    propertysearchrequest: Dict
    current_field: str
    completed_fields: List[str]
    is_complete: bool
    chat_history: List[Dict]  # Add chat history to state

# Worker state
class WorkerState(TypedDict):
    field: str  # Changed from section to field
    completed_fields: Annotated[list, operator.add]

class UserPreferencesSearch(BaseModel):
    user_preferences: UserPreferences
    search_parameters: PropertySearchRequest

# Default LLM
llm = ChatOpenAI(
    api_key=settings.GEMINI_API_KEY,
    base_url=settings.BASE_URL,
    model="gemini-2.0-flash"
)



# Structured output parser
def PreferenceGraph():
    # Worker: Extract preferences from input
    def extract_worker(state: State) -> State:
        """Extract preferences and search criteria from user input"""
        parser = JsonOutputParser(pydantic_object=UserPreferencesSearch)
        messages = state["messages"]
        current_preferences = state["userpreferences"]
        current_search_params = state["propertysearchrequest"]
        chat_history = state.get("chat_history", [])
        
        if not messages or not isinstance(messages[-1], HumanMessage):
            return state
            
        try:
            # Extract both user preferences and search parameters
            extraction_prompt = """You are an expert Australian real estate consultant and demographic analyst.
            Analyze the conversation to extract two types of information:

            1. Detailed user preferences with importance weights (0.0-1.0):
            {
                "Location": ["specific-location", weight],
                "Price": ["price-range", weight],
                "Size": ["size-requirements", weight],
                "Layout": ["layout-preferences", weight],
                "PropertyType": ["type-preference", weight],
                "Features": ["desired-features", weight],
                "Condition": ["condition-preference", weight],
                "Environment": ["environment-requirements", weight],
                "Style": ["style-preference", weight],
                "Quality": ["quality-requirements", weight],
                "Room": ["room-requirements", weight],
                "SchoolDistrict": ["school-requirements", weight],
                "Community": ["community-preferences", weight],
                "Transport": ["transport-requirements", weight],
                "Other": ["other-preferences", weight]
            }

            2. Specific search parameters:
            {
                "location": "STATE-SUBURB-POSTCODE",
                "suburb": "suburb name",
                "state": "state code",
                "postcode": number,
                "min_price": number or null,
                "max_price": number or null,
                "min_bedrooms": number or null,
                "property_type": "house/apartment/etc" or null
            }

            For location format:
            - Use STATE-SUBURB-POSTCODE format (e.g., "NSW-Chatswood-2067")
            - Break down into suburb, state, and postcode components
            
            For preferences:
            - Assign higher weights (0.8-1.0) to explicitly stated requirements
            - Use medium weights (0.4-0.7) for inferred preferences
            - Use lower weights (0.1-0.3) for contextual hints
            
            Return both objects in the format:
            {
                "user_preferences": {preference object},
                "search_parameters": {search parameters object}
            }
            """
            
            # Build conversation context 
            conversation = "\nConversation:\n"
            for msg in chat_history:
                conversation += f"{msg['role'].title()}: {msg['content']}\n"
            conversation += f"User: {messages[-1].content}\n"
            
            # Get LLM response
            response = llm.invoke([
                SystemMessage(content=extraction_prompt),
                HumanMessage(content=conversation)
            ])
            
            print("Conversation context:", conversation)
            print("LLM response:", response.content.strip())
            
            # Parse extracted preferences
            output = parser.invoke(response)
            # Extract the relevant portions and parse them
            new_search_params = output["search_parameters"]
            new_user_prefs = output["user_preferences"]
            print("Parsed search_params:", new_search_params)
            print("Parsed user_prefs:", new_user_prefs)
            
            # Update preferences, keeping existing values if not in new extraction
            merged_preferences = {**current_preferences}
            for key, value in new_user_prefs.items():
                    merged_preferences[key] = value
            print("Merged preferences:", merged_preferences)
                        
            merged_search_params = {**current_search_params}
            for key, value in new_search_params.items():
                    merged_search_params[key] = value
                    
            print("Merged merged_search_params:", merged_search_params)
            # Update state with merged values
            state["userpreferences"] = merged_preferences
            state["propertysearchrequest"] = merged_search_params
            print("State:", state)
        except Exception as e:
            print(f"Error in extract_worker: {e}")
        
        return state

    # Worker: Check if all required fields are present
    def check_worker(state: State) -> State:
        """Check if all required preferences are present"""
        search_params = state["propertysearchrequest"]
        required_fields = ["location", "max_price", "property_type"]
        
        # Check each required field
        missing = []
        for field in required_fields:
            if search_params.get(field) is None:
                missing.append(field)
        
        if missing:
            # Set the first missing field as current field
            state["current_field"] = missing[0]
            state["is_complete"] = False
            print(f"Missing fields: {missing}")
        else:
            state["is_complete"] = True
            print("All required fields collected:", search_params)
            
        return state

    # Worker: Generate follow-up question
    def question_worker(state: State) -> State:
        """Generate follow-up question for missing information"""
        current_field = state["current_field"]
        search_params = state["propertysearchrequest"]
        
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
            The user's current preferences: {search_params}
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
        self.graph = PreferenceGraph()
        self.chat_history = []  # Initialize chat history
        self.property_scraper = PropertyScraper()  # Initialize property scraper
    
    async def process_user_input(self, user_input: str, preferences: Dict = None, search_params: Dict = None) -> Tuple[str, Dict]:
        """Process user input through the graph"""
        # Add user message to chat history
        self.chat_history.append({"role": "user", "content": user_input})
        
        # Initialize state with chat history
        state = State(
            messages=[HumanMessage(content=user_input)],
            userpreferences=preferences or UserPreferences(),
            propertysearchrequest=search_params or PropertySearchRequest(),
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
        updated_preferences = final_state["userpreferences"]
        updated_search_params = final_state["propertysearchrequest"]
        # If we have complete preferences, search for properties
        # if final_state["is_complete"]:
        #     try:    
        #         # Search for properties using scraper
        #         properties = await self.property_scraper.search_properties(
        #             location=updated_preferences.get("location"),
        #             min_price=updated_preferences.get("min_price"),
        #             max_price=updated_preferences.get("max_price"),
        #             min_beds=updated_preferences.get("min_bedrooms"),
        #             property_type=updated_preferences.get("property_type"),
        #             max_results=5  # Limit results
        #         )
        #         print(properties)
        #         # Format property results into response
        #         if properties:
        #             property_summary = "\n\nHere are some properties that match your criteria:\n"
        #             for i, prop in enumerate(properties, 1):
        #                 property_summary += f"\n{i}. {prop['address']}"
        #                 if prop['price']:
        #                     property_summary += f"\nPrice: {prop['price']}"
        #                 property_summary += f"\nBedrooms: {prop['bedrooms']}"
        #                 if prop['property_type']:
        #                     property_summary += f"\nType: {prop['property_type']}"
        #                 property_summary += "\n"
                    
        #             last_message += property_summary
        #         else:
        #             last_message += "\n\nI couldn't find any properties matching your criteria at the moment."
                
        #     except Exception as e:
                # print(f"Error searching properties: {e}")
                # last_message += "\n\nI encountered an error while searching for properties."
        
        # Add assistant response to chat history
        self.chat_history.append({"role": "assistant", "content": last_message})
        
        return last_message, updated_preferences, updated_search_params