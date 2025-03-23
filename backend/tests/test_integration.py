import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json
from fastapi.testclient import TestClient

from app.main import app
from app.models import (
    PropertySearchResponse,
    FirestoreProperty,
    UserPreferences,
    PropertySearchRequest,
    PropertyBasicInfo,
    PropertyMedia,
    PropertyEvents,
    PropertyMetadata,
    AgentInfo
)
from app.services.image_processor import PropertyAnalysis, Style, Features, QualityAssessment, RoomAnalysis, Environment, QualityFactors

client = TestClient(app)

@pytest.fixture
def mock_property_search_response():
    """Sample property search response data"""
    return PropertySearchResponse(
        listing_id="test-listing-001",
        price="$850,000",
        address="123 Test Street, Sydney, NSW 2000",
        bedrooms="3",
        bathrooms="2",
        car_parks="1",
        property_type="House",
        land_size="300 sqm",
        inspection_date="Saturday, 10 June 2023",
        image_urls=["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
        agent_name="Test Agent"
    )

@pytest.fixture
def mock_firestore_property():
    """Sample FirestoreProperty object"""
    return FirestoreProperty(
        listing_id="test-listing-001",
        basic_info=PropertyBasicInfo(
            price_value=850000.0,
            price_is_numeric=True,
            full_address="123 Test Street, Sydney, NSW 2000",
            street_address="123 Test Street",
            suburb="Sydney",
            state="NSW",
            postcode="2000",
            bedrooms_count=3,
            bathrooms_count=2.0,
            car_parks="1",
            land_size="300 sqm",
            property_type="House"
        ),
        media=PropertyMedia(
            image_urls=["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
            main_image_url="https://example.com/img1.jpg"
        ),
        events=PropertyEvents(
            inspection_date="Saturday, 10 June 2023"
        ),
        agent=AgentInfo(
            agent_name="Test Agent"
        ),
        metadata=PropertyMetadata(),
        analysis={
            "style": {
                "architectural_style": "Modern",
                "era": "Contemporary",
                "design_elements": ["Open plan", "High ceilings"]
            },
            "features": {
                "interior": ["Hardwood floors", "Granite countertops"],
                "exterior": ["Deck", "Landscaped garden"],
                "notable_amenities": ["Swimming pool", "BBQ area"]
            },
            "quality_assessment": {
                "overall_score": 8.5,
                "condition": "Excellent",
                "maintenance_level": "Well maintained",
                "build_quality": "High quality",
                "environment": {
                    "privacy": "Good",
                    "noise_exposure": "Low",
                    "lighting_conditions": "Bright, natural light",
                    "sustainability_features": ["Solar panels"],
                    "road_proximity": "Low traffic",
                    "pole_or_line_of_sight": "No obstructions",
                    "land_flatness": "Mostly flat",
                    "greenery": "Well vegetated"
                },
                "factors": {
                    "positive": ["Good natural light", "Quality finishes"],
                    "negative": ["Small backyard"]
                }
            },
            "room_analysis": {
                "space_usage": "Efficient",
                "natural_light": "Abundant",
                "layout_quality": "Well designed"
            }
        }
    )

@pytest.fixture
def mock_property_analysis():
    """Sample PropertyAnalysis object"""
    return PropertyAnalysis(
        style=Style(
            architectural_style="Modern",
            era="Contemporary",
            design_elements=["Open plan", "High ceilings"]
        ),
        features=Features(
            interior=["Hardwood floors", "Granite countertops"],
            exterior=["Deck", "Landscaped garden"],
            notable_amenities=["Swimming pool", "BBQ area"]
        ),
        quality_assessment=QualityAssessment(
            overall_score=8.5,
            condition="Excellent",
            maintenance_level="Well maintained",
            build_quality="High quality",
            environment=Environment(
                privacy="Good",
                noise_exposure="Low",
                lighting_conditions="Bright, natural light",
                sustainability_features=["Solar panels"],
                road_proximity="Low traffic",
                pole_or_line_of_sight="No obstructions",
                land_flatness="Mostly flat",
                greenery="Well vegetated"
            ),
            factors=QualityFactors(
                positive=["Good natural light", "Quality finishes"],
                negative=["Small backyard"]
            )
        ),
        room_analysis=RoomAnalysis(
            space_usage="Efficient",
            natural_light="Abundant",
            layout_quality="Well designed"
        )
    )

@pytest.fixture
def mock_user_preferences():
    """Sample user preferences"""
    return {
        "budget_min": 700000,
        "budget_max": 900000,
        "bedrooms_min": 2,
        "location": ["Sydney"],
        "property_types": ["House", "Apartment"],
        "must_have_features": ["outdoor space", "parking"],
        "nice_to_have_features": ["pool", "modern kitchen"],
        "deal_breakers": ["busy road", "renovation required"]
    }

@pytest.fixture
def mock_search_params():
    """Sample search parameters"""
    return {
        "location": "Sydney",
        "suburb": "Sydney",
        "state": "NSW",
        "min_price": 700000,
        "max_price": 900000,
        "min_bedrooms": 2,
        "property_type": "House"
    }

# Integration Tests

class TestPropertySearchFlow:
    """Test the complete property search flow"""

    @pytest.fixture
    def mock_property_search_response(self):
        """Mock property search response data"""
        return PropertySearchResponse(
            listing_id="test-listing-001",
            price="$850,000",
            address="123 Test Street, Sydney, NSW 2000",
            bedrooms="3",
            bathrooms="2",
            car_parks="1",
            property_type="House",
            land_size="300 sqm",
            inspection_date="Saturday, 10 June 2023",
            image_urls=["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
            agent_name="Test Agent"
        )

    @pytest.fixture
    def mock_firestore_property(self):
        """Mock Firestore property data"""
        return FirestoreProperty(
            listing_id="test-listing-001",
            basic_info=PropertyBasicInfo(
                price_value=850000.0,
                price_is_numeric=True,
                full_address="123 Test Street, Sydney, NSW 2000",
                street_address="123 Test Street",
                suburb="Sydney",
                state="NSW",
                postcode="2000",
                bedrooms_count=3,
                bathrooms_count=2.0,
                car_parks="1",
                land_size="300 sqm",
                property_type="House"
            ),
            media=PropertyMedia(
                image_urls=["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
                main_image_url="https://example.com/img1.jpg",
                video_url=None
            ),
            agent=AgentInfo(
                agent_name="Test Agent",
                agency=None,
                contact_number=None,
                email=None
            ),
            events=PropertyEvents(
                inspection_date="Saturday, 10 June 2023",
                inspection_times=None,
                auction_date=None,
                listing_date=None,
                last_updated_date=None
            ),
            metadata=PropertyMetadata(
                created_at="2025-03-23T22:39:58.127648",
                updated_at="2025-03-23T22:39:58.127649",
                last_analysis_at=None,
                source="scraper",
                status="active"
            ),
            analysis={
                "style": {
                    "architectural_style": "Modern"
                },
                "features": {
                    "interior": ["Open plan kitchen", "Hardwood floors"],
                    "exterior": ["Balcony", "Garden"],
                    "notable_amenities": ["Near public transport", "Parking"]
                },
                "quality_assessment": {
                    "overall_score": 8.5,
                    "condition": "Excellent",
                    "maintenance_level": "Well maintained",
                    "build_quality": "High quality",
                    "environment": {
                        "privacy": "Good",
                        "noise_exposure": "Low",
                        "lighting_conditions": "Bright, natural light",
                        "sustainability_features": ["Solar panels"],
                        "road_proximity": "Low traffic"
                    }
                },
                "room_analysis": {
                    "space_usage": "Efficient",
                    "natural_light": "Abundant",
                    "layout_quality": "Well designed"
                }
            }
        )

    @pytest.fixture
    def mock_property_analysis(self):
        """Mock property analysis data"""
        return PropertyAnalysis(
            style={
                "architectural_style": "Modern",
                "era": "Contemporary",
                "design_elements": ["Open plan", "High ceilings"]
            },
            features={
                "interior": ["Hardwood floors", "Granite countertops"],
                "exterior": ["Deck", "Landscaped garden"],
                "notable_amenities": ["Swimming pool", "BBQ area"]
            },
            quality_assessment={
                "overall_score": 8.5,
                "condition": "Excellent",
                "maintenance_level": "Well maintained",
                "build_quality": "High quality",
                "environment": {
                    "privacy": "Good",
                    "noise_exposure": "Low",
                    "lighting_conditions": "Bright, natural light",
                    "sustainability_features": ["Solar panels"],
                    "road_proximity": "Low traffic",
                    "pole_or_line_of_sight": "No obstructions",
                    "land_flatness": "Mostly flat",
                    "greenery": "Well vegetated"
                },
                "factors": {
                    "positive": ["Good natural light", "Quality finishes"],
                    "negative": ["Small backyard"]
                }
            },
            room_analysis={
                "space_usage": "Efficient",
                "natural_light": "Abundant",
                "layout_quality": "Well designed"
            }
        )

    @pytest.fixture
    def mock_user_preferences(self):
        """Mock user preferences"""
        return {
            "budget_min": 700000,
            "budget_max": 900000,
            "bedrooms_min": 2,
            "location": ["Sydney"],
            "property_types": ["House", "Apartment"],
            "must_have_features": ["outdoor space", "parking"],
            "nice_to_have_features": ["pool", "modern kitchen"],
            "deal_breakers": ["busy road", "renovation required"]
        }

    @pytest.fixture
    def mock_search_params(self):
        """Mock search parameters"""
        return {
            "location": "Sydney",
            "suburb": "Sydney",
            "state": "NSW",
            "min_price": 700000,
            "max_price": 900000,
            "min_bedrooms": 2,
            "property_type": "House"
        }

    @patch("app.main.llm_service")
    @patch("app.main.search_properties")
    @patch("app.main.firestore_service")
    @patch("app.main.process_image")
    @patch("app.main.recommend_properties")
    async def test_complete_property_search_flow(
        self,
        mock_recommend,
        mock_process_image,
        mock_firestore,
        mock_search,
        mock_llm,
        mock_property_search_response,
        mock_firestore_property,
        mock_property_analysis,
        mock_user_preferences,
        mock_search_params
    ):
        """Test the complete property search flow from chat to recommendations"""
        # Setup mocks
        mock_llm.process_user_input = AsyncMock(return_value={
            "messages": [{"content": "Here are some properties that match your criteria"}],
            "userpreferences": mock_user_preferences,
            "propertysearchrequest": mock_search_params,
            "is_complete": True
        })

        mock_search.return_value = [mock_property_search_response]
        mock_firestore.save_property = AsyncMock(return_value="test-listing-001")
        mock_firestore.get_property = AsyncMock(return_value=mock_firestore_property)
        mock_process_image.return_value = mock_property_analysis
        mock_recommend.return_value = [mock_firestore_property]

        # Test the actual endpoint without mocking chat_endpoint
        chat_response = client.post(
            "/api/v1/chat",
            json={
                "user_input": "I want to buy a house in Sydney under $900k with 2+ bedrooms",
                "preferences": None,
                "search_params": None
            }
        )
        
        # Verify chat response
        assert chat_response.status_code == 200
        assert "response" in chat_response.json()

    @patch("app.main.llm_service")
    @patch("app.main.search_properties")
    @patch("app.main.firestore_service")
    async def test_property_search_flow_with_existing_analysis(
        self,
        mock_firestore,
        mock_search,
        mock_llm,
        mock_property_search_response,
        mock_firestore_property,
        mock_user_preferences,
        mock_search_params
    ):
        """Test the property search flow when analysis already exists"""
        # Setup mocks
        mock_llm.process_user_input = AsyncMock(return_value={
            "messages": [{"content": "Here are some properties that match your criteria"}],
            "userpreferences": mock_user_preferences,
            "propertysearchrequest": mock_search_params,
            "is_complete": True
        })

        mock_search.return_value = [mock_property_search_response]
        # Simulate that the property already has analysis
        mock_firestore_property.analysis = {
            "style": {
                "architectural_style": "Modern"
            },
            "features": {
                "interior": ["Open plan kitchen", "Hardwood floors"],
                "exterior": ["Balcony", "Garden"],
                "notable_amenities": ["Near public transport", "Parking"]
            }
        }
        mock_firestore.get_property = AsyncMock(return_value=mock_firestore_property)

        # Test the actual endpoint without mocking chat_endpoint
        chat_response = client.post(
            "/api/v1/chat",
            json={
                "user_input": "I want to buy a house in Sydney under $900k with 2+ bedrooms",
                "preferences": None,
                "search_params": None
            }
        )
        
        # Verify chat response
        assert chat_response.status_code == 200
        assert "response" in chat_response.json()

    @patch("app.main.llm_service")
    @patch("app.main.search_properties")
    @patch("app.main.firestore_service")
    @patch("app.main.process_image")
    async def test_property_search_flow_with_error_in_image_processing(
        self,
        mock_process_image,
        mock_firestore,
        mock_search,
        mock_llm,
        mock_property_search_response,
        mock_user_preferences,
        mock_search_params
    ):
        """Test error handling in the image processing step"""
        # Setup mocks
        mock_llm.process_user_input = AsyncMock(return_value={
            "messages": [{"content": "Here are some properties that match your criteria"}],
            "userpreferences": mock_user_preferences,
            "propertysearchrequest": mock_search_params,
            "is_complete": True
        })
        
        mock_search.return_value = [mock_property_search_response]
        mock_firestore.save_property = AsyncMock(return_value="test-listing-001")
        
        # Simulate error in image processing
        from fastapi import HTTPException
        mock_process_image.side_effect = HTTPException(
            status_code=500, 
            detail="Error processing images"
        )

        # Mock the chat endpoint
        with patch("app.main.chat_endpoint") as mock_chat_endpoint:
            mock_chat_endpoint.return_value = {
                "response": "Here are some properties that match your criteria",
                "preferences": mock_user_preferences,
                "search_params": mock_search_params
            }
            
            # Step 1: Chat with the assistant
            chat_response = client.post(
                "/api/v1/chat",
                json={
                    "user_input": "I want to buy a house in Sydney under $900k with 2+ bedrooms",
                    "preferences": None,
                    "search_params": None
                }
            )
        
        # Verify chat response
        assert chat_response.status_code == 200

    @patch("app.main.llm_service")
    async def test_property_search_flow_incomplete_preferences(self, mock_llm):
        """Test when the user preferences aren't complete yet"""
        # Setup mocks with proper message structure
        mock_llm.process_user_input = AsyncMock(return_value={
            "messages": [{"content": "Could you tell me more about what you're looking for?"}],
            "userpreferences": None,
            "propertysearchrequest": None,
            "is_complete": False  # Not complete
        })

        # Execute the chat request
        response = client.post(
            "/api/v1/chat",
            json={
                "user_input": "I want to buy a house",
                "preferences": None,
                "search_params": None
            }
        )
        
        # Verify
        assert response.status_code == 200
        assert "response" in response.json()
        assert response.json()["response"] == "Could you tell me more about what you're looking for?"
        assert response.json()["preferences"] is None

    @patch("app.main.llm_service")
    @patch("app.main.search_properties")
    async def test_property_search_flow_with_no_results(
        self,
        mock_search,
        mock_llm,
        mock_user_preferences,
        mock_search_params
    ):
        """Test the property search flow when no properties are found"""
        # Setup mocks
        mock_llm.process_user_input = AsyncMock(return_value={
            "messages": [{"content": "Here are some properties that match your criteria"}],
            "userpreferences": mock_user_preferences,
            "propertysearchrequest": mock_search_params,
            "is_complete": True
        })

        # Return empty results
        mock_search.return_value = []

        # Test the actual endpoint without mocking chat_endpoint
        chat_response = client.post(
            "/api/v1/chat",
            json={
                "user_input": "I want to buy a house in a very specific location",
                "preferences": None,
                "search_params": None
            }
        )
        
        # Verify chat response
        assert chat_response.status_code == 200
        assert "response" in chat_response.json()
        assert mock_search.called 