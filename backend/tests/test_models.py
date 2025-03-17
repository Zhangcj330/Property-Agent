import pytest
from datetime import datetime
from app.models import (
    PropertySearchResponse, 
    FirestoreProperty,
    PropertyBasicInfo,
    PropertyMedia,
    AgentInfo,
    PropertyEvents
)

# Test fixtures for different property scenarios
@pytest.fixture
def standard_property():
    """Standard property with all fields populated"""
    return PropertySearchResponse(
        listing_id="listing-16511207",
        price="$1,495,000",
        address="35 Samuel Street, Surry Hills, NSW 2010",
        bedrooms=3,
        bathrooms=1,
        car_parks=0,
        land_size="55㎡",
        property_type="House",
        inspection_date="Inspection Wed 05 Mar",
        image_urls=["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
        agent_name="Wiseberry Enmore"
    )

@pytest.fixture
def contact_agent_property():
    """Property with 'Contact agent' as price"""
    return PropertySearchResponse(
        listing_id="prop456",
        price="Contact agent",
        address="45 Park Avenue, South Yarra, VIC 3141",
        bedrooms="4",
        bathrooms="3",
        car_parks="1",
        land_size="350 sqm",
        property_type="Apartment",
        inspection_date="Sunday, 11 June 2023",
        image_urls=["https://example.com/img3.jpg"],
        agent_name="Jane Doe"
    )

@pytest.fixture
def price_range_property():
    """Property with price range"""
    return PropertySearchResponse(
        listing_id="prop789",
        price="$800,000 - $850,000",
        address="67 Beach Road, Brighton, VIC 3186",
        bedrooms="5",
        bathrooms="3.5",
        car_parks="3",
        land_size="600 sqm",
        property_type="House",
        inspection_date=None,
        image_urls=[],
        agent_name="Robert Brown"
    )

@pytest.fixture
def minimal_property():
    """Property with minimal information"""
    return PropertySearchResponse(
        listing_id="prop999",
        price="$500k",
        address="1/42 Smith St, Melbourne",
        bedrooms="1",
        bathrooms="1",
        car_parks=None,
        land_size=None,
        property_type=None,
        inspection_date=None,
        image_urls=None,
        agent_name=None
    )

# Tests for from_search_response method
def test_standard_property_conversion(standard_property):
    """Test conversion of a standard property"""
    firestore_property = FirestoreProperty.from_search_response(standard_property)
    
    # Check core identification
    assert firestore_property.listing_id == "listing-16511207"
    
    # Check basic info
    assert firestore_property.basic_info.price_value == 1495000
    assert firestore_property.basic_info.price_is_numeric == True
    assert firestore_property.basic_info.full_address == "35 Samuel Street, Surry Hills, NSW 2010"
    assert firestore_property.basic_info.street_address == "35 Samuel Street"
    assert firestore_property.basic_info.suburb == "Surry Hills"
    assert firestore_property.basic_info.state == "NSW"
    assert firestore_property.basic_info.postcode == "2010"
    assert firestore_property.basic_info.bedrooms_count == 3
    assert firestore_property.basic_info.bathrooms_count == 1
    assert firestore_property.basic_info.car_parks == "0"
    assert firestore_property.basic_info.land_size == "55㎡"
    assert firestore_property.basic_info.property_type == "House"
    
    # Check media
    assert len(firestore_property.media.image_urls) == 2
    assert firestore_property.media.main_image_url == "https://example.com/img1.jpg"
    
    # Check agent info
    assert firestore_property.agent.agent_name == "Wiseberry Enmore"
    
    # Check events
    assert firestore_property.events.inspection_date == "Inspection Wed 05 Mar"
    
    # Check metadata exists
    assert isinstance(firestore_property.metadata.created_at, datetime)
    assert isinstance(firestore_property.metadata.updated_at, datetime)

def test_contact_agent_price(contact_agent_property):
    """Test handling of 'Contact agent' price"""
    firestore_property = FirestoreProperty.from_search_response(contact_agent_property)
    
    # Check price handling
    assert firestore_property.basic_info.price_value is None
    assert firestore_property.basic_info.price_is_numeric == False

def test_price_range_extraction(price_range_property):
    """Test extraction of price from a range"""
    firestore_property = FirestoreProperty.from_search_response(price_range_property)
    
    # Should extract the first number from the range
    assert firestore_property.basic_info.price_value == 800000
    assert firestore_property.basic_info.price_is_numeric == True

def test_minimal_property(minimal_property):
    """Test conversion with minimal information"""
    firestore_property = FirestoreProperty.from_search_response(minimal_property)
    
    # Check basic info
    assert firestore_property.listing_id == "prop999"
    assert firestore_property.basic_info.price_value == 500
    assert firestore_property.basic_info.price_is_numeric == True
    
    # Check address parsing with minimal address
    assert firestore_property.basic_info.full_address == "1/42 Smith St, Melbourne"
    assert firestore_property.basic_info.street_address == "1/42 Smith St"
    assert firestore_property.basic_info.suburb == "Melbourne"
    
    # Check optional fields
    assert firestore_property.basic_info.car_parks is None
    assert firestore_property.basic_info.land_size is None
    assert firestore_property.basic_info.property_type is None
    assert firestore_property.events.inspection_date is None
    assert firestore_property.agent.agent_name is None

def test_round_trip_conversion(standard_property):
    """Test round-trip conversion from PropertySearchResponse to FirestoreProperty and back"""
    firestore_property = FirestoreProperty.from_search_response(standard_property)
    search_response = firestore_property.to_search_response()
    
    # Check that key fields are preserved (values may be formatted differently)
    assert search_response.listing_id == standard_property.listing_id
    assert search_response.address == standard_property.address
    assert int(search_response.bedrooms) == int(standard_property.bedrooms)
    assert float(search_response.bathrooms) == float(standard_property.bathrooms)
    assert search_response.car_parks == standard_property.car_parks
    assert search_response.land_size == standard_property.land_size
    assert search_response.property_type == standard_property.property_type
    assert search_response.inspection_date == standard_property.inspection_date
    assert search_response.image_urls == standard_property.image_urls
    assert search_response.agent_name == standard_property.agent_name 