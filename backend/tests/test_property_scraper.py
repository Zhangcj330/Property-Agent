import pytest
import asyncio
from app.services.property_scraper import PropertyScraper
from app.models import PropertySearchRequest
import logging

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def property_scraper():
    return PropertyScraper()

@pytest.mark.asyncio
async def test_search_properties_basic(property_scraper):
    """Test basic property search functionality"""
    # Define test search parameters
    search_params = PropertySearchRequest(
        location=["NSW-Chatswood-2067"],
        min_bedrooms=3,
        max_price=3500000,
        property_type=["House"]
    )
    
    logger.info("Starting property search test with params: %s", search_params)
    
    # Execute search
    results = await property_scraper.search_properties(search_params, max_results=5)
    
    # Log results
    logger.info("Search completed. Found %d properties", len(results))
    for idx, prop in enumerate(results):
        logger.info("Property %d: %s - %s", idx + 1, prop.address, prop.price)
    
    # Basic assertions
    assert isinstance(results, list)
    if results:  # If we found any properties
        first_property = results[0]
        logger.info("First property details: %s", first_property)
        assert first_property.address is not None
        assert first_property.price is not None
        assert first_property.bedrooms is not None

@pytest.mark.asyncio
async def test_search_properties_no_results(property_scraper):
    """Test search with parameters that should return no results"""
    # Use unrealistic search parameters
    search_params = PropertySearchRequest(
        location=["NSW-Chatswood-2067"],
        min_price=100000000,  # Unrealistically high price
        property_type=["House"]
    )
    
    logger.info("Starting no-results test with params: %s", search_params)
    
    results = await property_scraper.search_properties(search_params)
    
    logger.info("Search completed. Found %d properties", len(results))
    assert len(results) == 0

@pytest.mark.asyncio
async def test_search_url_building(property_scraper):
    """Test URL construction with different parameters"""
    test_cases = [
        {
            "params": PropertySearchRequest(
                location=["NSW-Chatswood-2067"],
                min_bedrooms=3,
                max_price=3500000,
                property_type=["House"]
            ),
            "expected_contains": [
                "3-bedrooms",
                "nsw-chatswood-2067",
                "3500000",
                "house"
            ]
        },
        {
            "params": PropertySearchRequest(
                location=["VIC-Melbourne-3000"],
                min_price=500000,
                max_price=1000000
            ),
            "expected_contains": [
                "vic-melbourne-3000",
                "500000",
                "1000000"
            ]
        }
    ]
    
    for case in test_cases:
        url = property_scraper._build_search_url(case["params"])
        logger.info("Generated URL: %s", url)
        for expected in case["expected_contains"]:
            assert expected in url.lower(), f"Expected {expected} in URL"

@pytest.mark.asyncio
async def test_parse_listing(property_scraper):
    """Test parsing a sample listing HTML"""
    # Create a minimal HTML structure for testing
    sample_html = """
    <span data-testid="listing-123">
        <p data-testid="property-card-title">$1,500,000</p>
        <h2>123 Test Street, Chatswood NSW 2067</h2>
        <div data-testid="bedroom"><span>3</span></div>
        <div data-testid="bathroom"><span>2</span></div>
        <div data-testid="carpark"><span>1</span></div>
    </span>
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(sample_html, 'html.parser')
    listing = soup.find('span', {'data-testid': 'listing-123'})
    
    result = property_scraper._parse_listing(listing)
    logger.info("Parsed listing result: %s", result)
    
    assert result is not None
    assert result.listing_id == "listing-123"
    assert result.price == "$1,500,000"
    assert "123 Test Street" in result.address
    assert result.bedrooms == "3"
    assert result.bathrooms == "2"
    assert result.car_parks == "1" 