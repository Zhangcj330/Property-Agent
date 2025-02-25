import streamlit as st
import requests
import json
from typing import Dict, List
import pandas as pd
import plotly.express as px

BACKEND_URL = "http://localhost:8000"

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_preferences' not in st.session_state:
        st.session_state.current_preferences = None
    if 'properties' not in st.session_state:
        st.session_state.properties = []

def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

def display_chat_message(role: str, content: str):
    with st.chat_message(role):
        st.write(content)

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

def process_chat_response(response: str, preferences: Dict):
    """Process the chat response and preferences"""
    with st.chat_message("assistant"):
        st.write(response)
        
        # If we have preferences, show them in an expander
        if preferences and any(preferences.values()):
            with st.expander("📋 I understood these preferences"):
                st.json(preferences)
            
            # If we have complete preferences, show property recommendations
            if all(k in preferences and preferences[k] for k in ['location', 'max_price', 'min_bedrooms']):
                show_recommendations(preferences)

def show_recommendations(preferences: Dict):
    """Show property recommendations based on preferences"""
    try:
        recommendations = requests.post(
            f"{BACKEND_URL}/recommend",
            json=preferences
        ).json()
        
        if recommendations:
            st.write(f"📍 I found {len(recommendations)} properties matching your criteria:")
            for prop in recommendations[:3]:  # Show top 3
                display_property_in_chat(prop)
        else:
            st.write("I couldn't find any properties matching exactly these criteria. Would you like to adjust your preferences?")
    except Exception as e:
        st.error("Sorry, I couldn't fetch property recommendations at the moment.")

def main():
    st.set_page_config(page_title="Property Finder Chat", layout="wide")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
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
        try:
            response = requests.post(
                f"{BACKEND_URL}/extract-preferences",
                params={"user_input": prompt}
            )
            
            if response.status_code == 200:
                chat_response = response.json()
                st.session_state.messages.append({"role": "assistant", "content": chat_response["response"]})
                with st.chat_message("assistant"):
                    st.write(chat_response["response"])
        
            else:
                st.error("Sorry, I'm having trouble understanding. Could you rephrase that?")
        except Exception as e:
            st.error("Sorry, I'm having technical difficulties. Please try again.")

    # First-time user help message
    if not st.session_state.messages:
        st.info("""
        👋 Hi! I'm your AI real estate assistant. I can help you:
        - Find properties based on your preferences
        - Answer questions about locations and property types
        - Provide property recommendations
        
        Try saying something like:
        - "I'm looking for a 3-bedroom house in San Francisco under $1.5M"
        - "What properties are available in Seattle?"
        - "Tell me about properties with a garden"
        """)

if __name__ == "__main__":
    main() 