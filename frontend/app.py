import streamlit as st
import requests
import json
from typing import Dict, List

BACKEND_URL = "http://localhost:8000"
v1_prefix = "/api/v1"

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'search_params' not in st.session_state:
        st.session_state.search_params = {
            "location": None,
            "suburb": None, 
            "state": None,
            "postcode": None,
            "min_price": None,
            "max_price": None,
            "min_bedrooms": None,
            "property_type": None
        }
    if 'preferences' not in st.session_state:
        st.session_state.preferences = {}


def display_property_in_chat(property: Dict):
    with st.chat_message("assistant"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if property.get('image_url'):
                st.image(property['image_url'], use_column_width=True)
        
        with col2:
            st.write(f"**{property['address']}, {property['city']}**")
            st.write(f"💰 ${property['price']:,.2f}")
            st.write(f"🛏️ {property['bedrooms']} bedrooms")
            st.write(f"🚿 {property['bathrooms']} bathrooms")
            st.write(f"📏 {property['square_footage']} sq ft")
            
            if st.button("More Details", key=f"more_{property['id']}"):
                with st.expander("Property Details"):
                    st.write(f"**Description:** {property['description']}")
                    st.write(f"**Property Type:** {property['property_type'].title()}")

def display_preferences():
    """Display current user preferences in the sidebar"""
    with st.sidebar:
        st.subheader("📋 Your Preferences")
        
        # Display search parameters
        if st.session_state.search_params:
            with st.expander("Search Parameters", expanded=True):
                for key, value in st.session_state.search_params.items():
                    if value is not None and key != 'must_have_features':
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                # Display features separately if they exist
                if st.session_state.search_params.get('must_have_features'):
                    st.write("**Must Have Features:**")
                    for feature in st.session_state.search_params['must_have_features']:
                        st.write(f"- {feature}")
        
        # Display detailed preferences with weights
        if st.session_state.preferences:
            with st.expander("Detailed Preferences", expanded=False):
                for category, details in st.session_state.preferences.items():
                    if isinstance(details, list) and len(details) >= 2:
                        preference, weight = details[0], details[1]
                        # Create a visual indicator of importance
                        importance = "🔴" if weight > 0.7 else "🟠" if weight > 0.4 else "🟡"
                        st.write(f"**{category}:** {preference} {importance}")
                    elif details:
                        st.write(f"**{category}:** {details}")
        
        # Add a button to clear preferences
        if st.button("Clear All Preferences"):
            st.session_state.preferences = {}
            st.session_state.search_params = {
                "location": None,
                "suburb": None, 
                "state": None,
                "postcode": None,
                "min_price": None,
                "max_price": None,
                "min_bedrooms": None,
                "property_type": None
            }
            st.rerun()

def show_sidebar():
    """Display and handle sidebar filters"""
    with st.sidebar:
        st.header("🔍 Property Filters")
        
        # Display current preferences first
        display_preferences()
        
        st.markdown("---")
        st.subheader("Adjust Filters")
        
        # Location filter
        location = st.text_input(
            "Location", 
            value=st.session_state.search_params.get('location', ''),
            placeholder="Enter city or area"
        )
        
        # Price range
        st.write("Price Range ($)")
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input(
                "Min Price",
                min_value=0,
                step=50000,
                value=st.session_state.search_params.get('min_price', 0)
            )
        with col2:
            max_price = st.number_input(
                "Max Price",
                min_value=0,
                step=50000,
                value=st.session_state.search_params.get('max_price', 0)
            )
        
        # Bedrooms
        min_beds = st.number_input(
            "Minimum Bedrooms",
            min_value=0,
            value=st.session_state.search_params.get('min_beds', 0)
        )
        
        # Property Type
        property_type = st.multiselect(
            "Property Type",
            options=['Any', 'house', 'apartment', 'condo', 'townhouse'],
            default=st.session_state.search_params.get('property_type', [])
        )
        
        # Features
        features = st.multiselect(
            "Must-Have Features",
            options=['Garage', 'Garden', 'Pool', 'Balcony', 'Parking'],
            default=st.session_state.search_params.get('must_have_features', [])
        )
        
        # Apply filters button
        if st.button("Apply Filters"):
            # Update preferences
            st.session_state.search_params.update({
                'location': location,
                'min_price': min_price if min_price > 0 else None,
                'max_price': max_price if max_price > 0 else None,
                'min_beds': min_beds if min_beds > 0 else None,
                'property_type': property_type if property_type else None,
                'must_have_features': features
            })
            st.rerun()

def process_chat_message(message: str):
    """Process chat message and get response from backend"""
    try:
        # Send both message and current preferences to backend
        response = requests.post(
            f"{BACKEND_URL}/{v1_prefix}/chat",
            json={
                "user_input": message,
                "preferences": st.session_state.preferences,
                "search_params": st.session_state.search_params
            }
        )
        
        if response.status_code == 200:
            chat_response = response.json()
            # Update preferences with any new information
            if chat_response.get("preferences"):
                st.session_state.preferences.update(chat_response["preferences"])
            if chat_response.get("search_params"):
                st.session_state.search_params.update(chat_response["search_params"])
                
            # Add a visual indicator that preferences were updated
            if (chat_response.get("preferences") or chat_response.get("search_params")):
                st.sidebar.success("✅ Your preferences have been updated!")

            return chat_response["response"]
        else:
            return "Sorry, I'm having trouble understanding. Could you rephrase that?"
    except Exception as e:
        print(f"Error in process_chat_message: {e}")
        return "Sorry, I'm having technical difficulties. Please try again."

def main():
    st.set_page_config(page_title="Property Finder Chat", layout="wide")
    init_session_state()
    
    # Show sidebar with filters and preferences
    show_sidebar()
    
    # Main chat interface
    st.title("🏠 Property Finder Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("How can I help you find your perfect property?"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response from backend
        response = process_chat_message(prompt)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

    # First-time user help message
    if not st.session_state.messages:
        st.info("""
        👋 Hi! I'm your AI real estate assistant. I can help you:
        - Find properties based on your preferences
        - Answer questions about locations and property types
        - Provide property recommendations
        
        Try saying something like:
        - "I'm looking for a 3-bedroom house in Sydney under $1.5M"
        - "What properties are available in Melbourne?"
        - "Tell me about properties with a garden in Brisbane"
        """)

if __name__ == "__main__":
    main()