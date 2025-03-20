import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.messages import HumanMessage

# Import your agent and models
from app.models import PropertySearchResponse, UserPreferences, UserPreference, FirestoreProperty
from app.PorpertyAgent.agent import agent, search_properties, analyze_property_images, get_property_recommendations
from app.services.image_processor import PropertyAnalysis

# Detailed mock property data provided
MOCK_PROPERTIES = [
    PropertySearchResponse(
        listing_id='listing-16525759', 
        price='Auction - Contact Agent', 
        address='43 De Villiers Avenue, Chatswood, NSW 2067', 
        bedrooms='5', 
        bathrooms='2', 
        car_parks='2', 
        land_size='1113㎡', 
        property_type='House', 
        inspection_date='Auction Sat 10 May', 
        image_urls=[
            'https://view.com.au/viewstatic/images/listing/5-bedroom-house-in-chatswood-nsw-2067/500-w/16525759-1-382876A.jpg', 
            'https://view.com.au/viewstatic/images/listing/5-bedroom-house-in-chatswood-nsw-2067/500-w/16525759-2-D6CF276.jpg', 
            'https://view.com.au/viewstatic/images/listing/5-bedroom-house-in-chatswood-nsw-2067/500-w/16525759-3-6A1C4CF.jpg'
        ], 
        agent_name='Ray White Ay Realty Chatswood'
    ), 
    PropertySearchResponse(
        listing_id='listing-16004086', 
        price='For Sale - Off Market Opportunity', 
        address='119 Greville Street, Chatswood, NSW 2067', 
        bedrooms='4', 
        bathrooms='2', 
        car_parks=None, 
        land_size='743㎡', 
        property_type='House', 
        inspection_date=None, 
        image_urls=[
            'https://view.com.au/viewstatic/images/listing/4-bedroom-house-in-chatswood-nsw-2067/500-w/16004086-1-7A10629.jpg', 
            'https://view.com.au/viewstatic/images/listing/4-bedroom-house-in-chatswood-nsw-2067/500-w/16004086-2-6B11C5C.jpg', 
            'https://view.com.au/viewstatic/images/listing/4-bedroom-house-in-chatswood-nsw-2067/500-w/16004086-13-D5F66E5.jpg'
        ], 
        agent_name='Ray White Ay Realty Chatswood'
    ), 
    PropertySearchResponse(
        listing_id='listing-16522764', 
        price='Auction guide: $3,500,000', 
        address='54 Highfield Road, Lindfield, NSW 2070', 
        bedrooms='4', 
        bathrooms='3', 
        car_parks='2', 
        land_size='1372㎡', 
        property_type='House', 
        inspection_date='Auction Sat 29 Mar', 
        image_urls=[
            'https://view.com.au/viewstatic/images/listing/4-bedroom-house-in-lindfield-nsw-2070/500-w/16522764-1-E27E12E.jpg', 
            'https://view.com.au/viewstatic/images/listing/4-bedroom-house-in-lindfield-nsw-2070/500-w/16522764-2-461120D.jpg', 
            'https://view.com.au/viewstatic/images/listing/4-bedroom-house-in-lindfield-nsw-2070/500-w/16522764-15-8177509.jpg'
        ], 
        agent_name='Ray White Upper North Shore'
    ), 
    PropertySearchResponse(
        listing_id='listing-16483614', 
        price='Auction - Guide $3,500,000', 
        address='103 Penshurst Street, Willoughby, NSW 2068', 
        bedrooms='4', 
        bathrooms='2', 
        car_parks='4', 
        land_size='919.80㎡', 
        property_type='House', 
        inspection_date='Auction Sat 22 Mar', 
        image_urls=[
            'https://view.com.au/viewstatic/images/listing/4-bedroom-house-in-willoughby-nsw-2068/500-w/16483614-1-544D62A.jpg', 
            'https://view.com.au/viewstatic/images/listing/4-bedroom-house-in-willoughby-nsw-2068/500-w/16483614-2-65EDD5F.jpg', 
            'https://view.com.au/viewstatic/images/listing/4-bedroom-house-in-willoughby-nsw-2068/500-w/16483614-14-CE9A0B7.png'
        ], 
        agent_name='Raine & Horne Lower North Shore'
    )
]

@pytest.fixture
def mock_search_service():
    """Mock only the property search service"""
    with patch('app.services.property_scraper.PropertyScraper.search_properties', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = MOCK_PROPERTIES
        yield mock_search

@pytest.fixture
def mock_image_processor():
    """Mock the image processor service"""
    with patch('app.services.image_processor.ImageProcessor.analyze_property_image', new_callable=AsyncMock) as mock_analyze:
        # Need to create Style, Features, QualityAssessment, and RoomAnalysis
        from app.services.image_processor import Style, Features, QualityAssessment, RoomAnalysis, Environment, QualityFactors
        
        mock_analyze.return_value = PropertyAnalysis(
            style=Style(
                architectural_style="Modern contemporary",
                era="Contemporary",
                design_elements=["Open floor plan", "Large windows", "Clean lines"]
            ),
            features=Features(
                interior=["Hardwood floors", "Natural light", "Modern kitchen", "High ceilings"],
                exterior=["Swimming pool", "Covered patio", "Landscaped garden"],
                notable_amenities=["Pool", "Home theater", "Walk-in closets"]
            ),
            quality_assessment=QualityAssessment(
                overall_score=9.0,
                condition="Excellent",
                maintenance_level="High",
                build_quality="Premium",
                environment=Environment(
                    privacy="Good",
                    noise_exposure="Low",
                    lighting_conditions="Excellent",
                    sustainability_features=["Solar panels", "Double glazing"],
                    road_proximity="Moderate",
                    pole_or_line_of_sight="Clear",
                    land_flatness="Mostly flat",
                    greenery="Abundant"
                ),
                factors=QualityFactors(
                    positive=["Well maintained", "Premium materials", "Modern design"],
                    negative=["Some wear on exterior paint"]
                )
            ),
            room_analysis=RoomAnalysis(
                space_usage="Efficient",
                natural_light="Abundant",
                layout_quality="Excellent"
            )
        )
        yield mock_analyze

@pytest.fixture
def mock_recommendation_service():
    """Mock the recommendation service"""
    with patch('app.services.recommender.PropertyRecommender.get_recommendations', new_callable=AsyncMock) as mock_recommend:
        # Import necessary models
        from app.models import PropertyBasicInfo, PropertyMedia, AgentInfo, PropertyEvents, PropertyMetadata
        
        # Return the same properties but with a "recommendation_reason" field
        mock_firestore_properties = []
        for prop in MOCK_PROPERTIES:
            # Create the basic info object
            basic_info = PropertyBasicInfo(
                price_value=3500000.0 if "3,500,000" in prop.price else None,
                price_is_numeric="3,500,000" in prop.price,
                full_address=prop.address,
                street_address=prop.address.split(',')[0] if ',' in prop.address else prop.address,
                suburb=prop.address.split(',')[1].strip() if ',' in prop.address and len(prop.address.split(',')) > 1 else None,
                state="NSW",
                postcode=prop.address.split('NSW')[1].strip() if 'NSW' in prop.address else None,
                bedrooms_count=int(prop.bedrooms) if prop.bedrooms and prop.bedrooms.isdigit() else None,
                bathrooms_count=float(prop.bathrooms) if prop.bathrooms and prop.bathrooms.isdigit() else None,
                car_parks=prop.car_parks,
                land_size=prop.land_size,
                property_type=prop.property_type
            )
            
            # Create media object
            media = PropertyMedia(
                image_urls=prop.image_urls,
                main_image_url=prop.image_urls[0] if prop.image_urls and len(prop.image_urls) > 0 else None
            )
            
            # Create agent info
            agent = AgentInfo(
                agent_name=prop.agent_name
            )
            
            # Create events
            events = PropertyEvents(
                inspection_date=prop.inspection_date
            )
            
            # Create a FirestoreProperty with all required fields
            mock_firestore_properties.append(FirestoreProperty(
                listing_id=prop.listing_id,
                basic_info=basic_info,
                media=media,
                agent=agent,
                events=events,
                metadata=PropertyMetadata(),
                analysis={
                    "architectural_style": "Modern contemporary with open floor plan",
                    "interior_features": "Spacious rooms with hardwood floors and natural light",
                    "exterior_features": "Well-maintained landscaping with pool",
                    "quality_score": 8,
                    "condition": "Excellent",
                    "renovation_potential": "Low - already renovated",
                    "neighborhood_impression": "Quiet, family-friendly area",
                    "recommendation_reason": "This property matches your preference for a modern home with quality finishes."
                }
            ))
        mock_recommend.return_value = mock_firestore_properties
        yield mock_recommend

@pytest.mark.asyncio
async def test_search_properties(mock_search_service):
    """Test that the agent can search for properties using mocked search service"""
    
    # Create test query
    test_query = "Find houses in Chatswood with at least 4 bedrooms"
    messages = [HumanMessage(content=test_query)]
    
    # Run the agent
    result = await agent.ainvoke({"messages": messages})
    
    # Verify search was called
    mock_search_service.assert_called_once()
    
    # Check if all properties appear in the response
    response_text = " ".join([msg.content for msg in result["messages"]])
    
    # The agent might format the response in different ways, so we just check for key details
    assert "Chatswood" in response_text
    
    # Check for listing IDs or addresses to verify properties are included
    assert any(id in response_text for id in ["listing-16525759", "listing-16004086"])
    assert any(street in response_text for street in ["De Villiers", "Greville"])
    
    # Verify that the agent found properties 
    assert any(term in response_text.lower() for term in ["found", "properties", "results", "houses"])

@pytest.mark.asyncio
async def test_image_analysis(mock_image_processor):
    """Test that the agent can analyze property images using mocked image processor"""
    
    # Create test query specifically for image analysis
    # Using a real image URL that should work
    test_query = "Analyze these property images: https://view.com.au/viewstatic/images/listing/5-bedroom-house-in-chatswood-nsw-2067/500-w/16525759-1-382876A.jpg"
    messages = [HumanMessage(content=test_query)]
    
    # Run the agent with mocked image analysis
    result = await agent.ainvoke({"messages": messages})
    
    # Verify mock was called
    mock_image_processor.assert_called_once()
    
    # Check if analysis details appear in the response
    response_text = " ".join([msg.content for msg in result["messages"]])
    
    # Test that analysis terms are in the response
    assert any(term in response_text.lower() for term in ["style", "architectural", "quality", "features"])

@pytest.mark.asyncio
async def test_autonomous_image_processing(mock_search_service, mock_image_processor):
    """Test that the agent autonomously processes images after property search using mocked services"""
    
    # Create a query that should trigger both property search and image analysis
    test_query = "Find houses in Chatswood with at least 4 bedrooms and analyze their images"
    messages = [HumanMessage(content=test_query)]
    
    # Run the agent with mocked search and image analysis
    result = await agent.ainvoke({"messages": messages})
    
    # Verify search was called
    mock_search_service.assert_called_once()
    
    # Combine all messages for analysis
    response_text = " ".join([msg.content for msg in result["messages"]])
    
    # Check if we have any property details in the response
    assert "Chatswood" in response_text
    assert any(id in response_text for id in ["listing-16525759", "listing-16004086"])
    
    # Verify that the mock_image_processor was called
    assert mock_image_processor.call_count > 0
    
    # Since the format of the response might vary, we check that the response
    # contains either some general information about properties or analysis results
    assert any(term in response_text.lower() for term in [
        "found", "properties", "results", "chatswood", 
        "image", "analyzed", "analysis", "photos", "picture"
    ])

@pytest.mark.asyncio
async def test_full_recommendation_flow(mock_search_service, mock_image_processor, mock_recommendation_service):
    """Test complete flow of search, image analysis, and recommendations using fully mocked services"""
    
    # Create a query for the complete flow
    test_query = """
    Find houses in Chatswood with at least 4 bedrooms,
    analyze the images, and recommend the best options for a family
    looking for a modern, quiet home with good quality finishes
    """
    
    messages = [HumanMessage(content=test_query)]
    
    # Run the agent - this should trigger search, analysis, and recommendations
    result = await agent.ainvoke({"messages": messages})
    
    # Verify search was called
    mock_search_service.assert_called_once()
    
    # Combine all messages for testing
    response_text = " ".join([msg.content for msg in result["messages"]])
    
    # Verify property search results
    assert "Chatswood" in response_text
    assert any(address in response_text for address in ["De Villiers Avenue", "Greville Street"])
    
    # Check for recommendation terms
    if "recommend" in response_text.lower():
        # Check for reasoning related to preferences
        assert any(term in response_text.lower() for term in ["modern", "quiet", "quality"])
    
    # Ensure at least one specific property is mentioned
    assert any(address in response_text for address in [
        "De Villiers Avenue", 
        "Greville Street", 
        "Highfield Road", 
        "Penshurst Street"
    ])

@pytest.mark.asyncio
async def test_recommendation_with_specific_preferences(mock_search_service, mock_image_processor, mock_recommendation_service):
    """Test recommendations with detailed user preferences using fully mocked services"""
    
    # First search for properties to establish context
    initial_query = "Find houses in Chatswood with at least 4 bedrooms"
    messages = [HumanMessage(content=initial_query)]
    await agent.ainvoke({"messages": messages})
    
    # Then request recommendations with specific preferences
    recommendation_query = """
    Based on these properties, please recommend the best option for me.
    I'm looking for:
    1. Modern contemporary style
    2. Excellent natural lighting
    3. A quiet neighborhood
    4. Good space for entertaining
    5. Energy efficient features
    """
    
    # Add recommendation query to existing conversation
    messages.append(HumanMessage(content=recommendation_query))
    
    # Run the agent with real recommendation service
    result = await agent.ainvoke({"messages": messages})
    
    # Analyze the response
    response_text = " ".join([msg.content for msg in result["messages"]])
    
    # Check for preference matching in the response
    assert any(preference in response_text.lower() for preference in [
        "modern", "contemporary", "natural light", "quiet", "entertaining", "energy"
    ])
    
    # Check for specific property mention
    assert any(address in response_text for address in [
        "De Villiers Avenue", 
        "Greville Street", 
        "Highfield Road", 
        "Penshurst Street"
    ])

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
