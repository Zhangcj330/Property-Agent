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
    current_missing_field: str
    is_complete: bool
    chat_history: List[Dict]  # Add chat history to state
    has_ambiguities: Optional[bool]
    ambiguities: Optional[List[Dict]]


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
            You are specializing in clarifying client requirements.
            Analyze the conversation to extract two types of information and make reasonable inferences about user preferences and search parameters.
            
            Rules for making inferences:
            1. Only make logical inferences based on clear contextual clues
            2. Assign lower confidence weights (0.1-0.3) to inferred preferences
            3. Don't override any explicitly stated preferences
            4. Consider lifestyle implications of stated preferences

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
            
            # Get LLM response
            response = llm.invoke([
                SystemMessage(content=extraction_prompt),
                HumanMessage(content=conversation)
            ])
            
            # print("Conversation context:", conversation)
            # print("LLM response:", response.content.strip())
            
            # Parse extracted preferences
            output = parser.invoke(response)
            # Extract the relevant portions and parse them
            new_search_params = output["search_parameters"]
            new_user_prefs = output["user_preferences"]

            # Update preferences, keeping existing values if not in new extraction
            merged_preferences = {**current_preferences}
            for key, value in new_user_prefs.items():
                    merged_preferences[key] = value
            # print("Merged preferences:", merged_preferences)
                        
            merged_search_params = {**current_search_params}
            for key, value in new_search_params.items():
                    merged_search_params[key] = value
                    
            # print("Merged search params:", merged_search_params)
            # Update state with merged values
            state["userpreferences"] = merged_preferences
            state["propertysearchrequest"] = merged_search_params
        except Exception as e:
            print(f"Error in extract_worker: {e}")
        
        
        return state

    # Worker: Check if all required fields are present
    def check_worker(state: State) -> State:
        """Check if all required preferences are present"""
        required_fields = ["location", "max_price", "property_type"]
        required_preferences = ["style", "layout", "community"]
        # Check each required field
        missing = []
        for field in required_fields:
            if state["propertysearchrequest"].get(field) is None:
                missing.append(field)

        for field in required_preferences:
            if state["userpreferences"].get(field) is None:
                missing.append(field)

        if missing:
            # Set the first missing field as current field
            state["current_missing_field"] = missing[0]
            state["has_missing_fields"] = True
            # print(f"Missing fields: {missing}")
        else:
            state["is_complete"] = True
            # print("All required fields collected")
            
        return state

    # Worker: Generate follow-up question
    def question_worker(state: State) -> State:
        """Generate follow-up question for missing information"""
        current_missing_field = state["current_missing_field"]
        search_params = state["propertysearchrequest"]
        chat_history = state.get("chat_history", [])
        
        # Build conversation context
        conversation = "\nPrevious conversation:\n"
        for msg in chat_history:
            conversation += f"{msg['role'].title()}: {msg['content']}\n"
        
        # Generate natural follow-up question
        response = llm.invoke([
            SystemMessage(content="""You are a knowledgeable and empathetic Australian real estate agent with years of experience.
            Your goal is to help users find their ideal property by gathering information and providing expert guidance.
            
            Guidelines:
            - Be conversational and empathetic
            - Ask one clear question at a time
            - If the user seems undecided after 2-3 exchanges about the same topic:
                * Provide market insights or suggestions based on their context
                * Share relevant examples or alternatives
                * Explain trade-offs between different options
                * Make professional recommendations based on their needs
            
            Examples of helpful suggestions:
            - For location: "Given your budget and preference for good schools, I'd suggest considering suburbs like X or Y. They offer great value and top-rated schools."
            - For price: "In the current market, properties in this area typically range from $X to $Y. Based on your requirements, I'd recommend a budget of around $Z."
            - For property type: "Since you're looking for low maintenance and good investment potential, a modern apartment in X area could be ideal."
            
            Current missing field: {current_missing_field}
            Current search parameters: {search_params}
            
            Generate either:
            1. A natural follow-up question to get missing information, or
            2. If the user seems undecided, provide expert guidance with specific suggestions."""),
            HumanMessage(content=f"""Based on the conversation:
            {conversation}
            
            We need to know about: {current_missing_field}
            Current search parameters: {search_params}
            
            The user has had {len(chat_history)} exchanges. If they seem undecided, provide expert guidance.
            Otherwise, ask a natural follow-up question to get the missing information.""")
        ])
        
        # Add question to messages
        state["messages"].append(AIMessage(content=response.content))
        return state

    # Worker: Infer preferences from context
    def inference_worker(state: State) -> State:
        """Infer additional preferences and search parameters from conversation context"""
        current_preferences = state["userpreferences"]
        current_search_params = state["propertysearchrequest"]
        chat_history = state.get("chat_history", [])
        
        if not chat_history:
            return state
            
        try:
            inference_prompt = """You are an expert real estate consultant with deep knowledge of Australian property market trends.
            Analyze the conversation to make reasonable inferences about user preferences and search parameters.
            
            Rules for making inferences:
            1. Only make logical inferences based on clear contextual clues
            2. Assign lower confidence weights (0.1-0.3) to inferred preferences
            3. Don't override any explicitly stated preferences
            4. Consider lifestyle implications of stated preferences
            
            Examples of valid inferences:
            - If user mentions "family", infer interest in multiple bedrooms
            - If user mentions "commute to CBD", infer transport requirements
            - If user mentions "quiet", infer preference for residential areas
            - If price range is high, infer quality expectations
            
            Return only reasonably confident inferences in the format:
            {
                "user_preferences": {
                    "Location": ["inferred-location", weight],
                    "Price": ["inferred-price-range", weight],
                    ... (other preference categories)
                },
                "search_parameters": {
                    "location": "inferred-location" or null,
                    "min_price": number or null,
                    "max_price": number or null,
                    "min_bedrooms": number or null,
                    "property_type": "inferred-type" or null
                }
            }
            """
            
            # Build conversation context
            conversation = "\nConversation context:\n"
            for msg in chat_history:
                conversation += f"{msg['role'].title()}: {msg['content']}\n"
            
            # Get inferences from LLM
            response = llm.invoke([
                SystemMessage(content=inference_prompt),
                HumanMessage(content=f"""Based on this conversation:
                {conversation}
                
                Current preferences: {current_preferences}
                Current search parameters: {current_search_params}
                
                What reasonable inferences can you make about their preferences and requirements?""")
            ])
            
            # Parse inferred preferences
            parser = JsonOutputParser(pydantic_object=UserPreferencesSearch)
            inferred = parser.invoke(response)
            
            # Merge inferences with existing preferences (only if not already set)
            merged_preferences = {**current_preferences}
            for key, value in inferred["user_preferences"].items():
                if key not in merged_preferences or merged_preferences[key][0] is None:
                    merged_preferences[key] = value
            
            # Merge inferred search parameters (only if not already set)
            merged_search_params = {**current_search_params}
            for key, value in inferred["search_parameters"].items():
                if key not in merged_search_params or merged_search_params[key] is None:
                    merged_search_params[key] = value
            
            # Update state with merged values
            state["userpreferences"] = merged_preferences
            state["propertysearchrequest"] = merged_search_params
            
            # print("Inferred preferences:", inferred["user_preferences"])
            # print("Inferred search params:", inferred["search_parameters"])
            
        except Exception as e:
            print(f"Error in inference_worker: {e}")
        
        return state

    # Worker: Check for ambiguities in user preferences
    def ambiguity_worker(state: State) -> State:
        """Identify and resolve ambiguities in user preferences and search parameters"""
        current_preferences = state["userpreferences"]
        current_search_params = state["propertysearchrequest"]
        chat_history = state.get("chat_history", [])
        
        if not chat_history:
            return state
            
        try:
            ambiguity_prompt = """As a Real Estate Advisor AI, your primary role is to identify and clarify ambiguities 
            or contradictions in the client's stated preferences or requirements. 

            Your goal is to avoid unnecessary probing into overly detailed specifics unless explicitly prompted by the 
            client's responses. Aim to provide concise and relevant assistance, ensuring an efficient and comfortable interaction.

            If no significant ambiguities are found, return {"ambiguities": [], "has_ambiguities": false}

            Return only significant ambiguities (max 2) in JSON format:          
                {
                "ambiguities": [
                    {
                        "type": "contradiction|vagueness|unrealistic",
                        "description": "Description of the ambiguity",
                        "importance": "high|medium|low",
                        "clarification_question": "Suggested question to resolve ambiguity"
                    }
                ],
                "has_ambiguities": true|false
            }
            """
            
            # Build conversation context
            conversation = "\nConversation context:\n"
            for msg in chat_history:
                conversation += f"{msg['role'].title()}: {msg['content']}\n"
            
            # Get ambiguity analysis from LLM
            response = llm.invoke([
                SystemMessage(content=ambiguity_prompt),
                HumanMessage(content=f"""Based on this conversation:
                {conversation}
                
                Identify any ambiguities or contradictions in the user's requirements:""")
            ])
            
            # Parse ambiguity analysis
            parser = JsonOutputParser()
            analysis = parser.invoke(response)
            
            # Store ambiguity information in state
            state["has_ambiguities"] = analysis.get("has_ambiguities", False)
            state["ambiguities"] = analysis.get("ambiguities", [])
            
            # print(f"Ambiguity check result: has_ambiguities={state['has_ambiguities']}")
            # if state["has_ambiguities"]:
            #     print(f"Found ambiguities: {state['ambiguities']}")
            
        except Exception as e:
            print(f"Error in ambiguity_worker: {e}")
            state["has_ambiguities"] = False
            state["ambiguities"] = []
        
        return state

    # Worker: Generate ambiguity clarification question
    def ambiguity_question_worker(state: State) -> State:
        """Generate follow-up question to clarify ambiguities"""
        ambiguities = state.get("ambiguities", [])
        
        if not ambiguities:
            return state
            
        # Get the most important ambiguity
        ambiguity = ambiguities[0]
        clarification_question = ambiguity["clarification_question"]
        
        # Add clarification question to messages
        state["messages"].append(AIMessage(content=clarification_question))
        
        # Add to chat history
        if "chat_history" in state:
            state["chat_history"].append({"role": "assistant", "content": clarification_question})
        
        # print(f"Asking for clarification: {clarification_question}")
        
        return state

    # Create the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("extract", extract_worker)
    # workflow.add_node("infer", inference_worker)
    workflow.add_node("ambiguity", ambiguity_worker)
    workflow.add_node("ambiguity_question", ambiguity_question_worker)  # Add new worker
    workflow.add_node("check", check_worker)
    workflow.add_node("question", question_worker)
    
    # Update edges
    workflow.add_edge(START, "extract")
    # workflow.add_edge("extract", "infer")
    workflow.add_edge("extract", "ambiguity")
    
    # Add conditional edge from ambiguity worker
    workflow.add_conditional_edges(
        "ambiguity",
        lambda x: "ambiguity_question" if x.get("has_ambiguities", False) else "check"
    )
    
    # Connect ambiguity question to END
    workflow.add_edge("ambiguity_question", END)
    
    # Connect check to either END or question
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
            missing_field=None,
            is_complete=False,
            chat_history=self.chat_history
        )
        
        # Run the graph
        final_state = self.graph.invoke(state)
        # print("Final state:", final_state)
        
        # If we have complete preferences, search for properties
        if not final_state["is_complete"]:
            last_message = final_state["messages"][-1].content if final_state["messages"] else ""
            self.chat_history.append({"role": "assistant", "content": last_message})
        
        return final_state