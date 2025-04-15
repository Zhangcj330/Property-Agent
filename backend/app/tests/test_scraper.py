"""
Tests for the PropertyScraper class
"""
import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.app.services.property_scraper import PropertyScraper
from backend.app.models import PropertySearchRequest

class TestPropertyScraper(unittest.TestCase):
    def setUp(self):
        self.scraper = PropertyScraper()
    
    def test_build_search_url_with_basic_params(self):
        # Test with minimum required parameters
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"],
            min_price=500000,
            max_price=1000000,
            property_type=["house"]
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("loc=nsw-sydney-2000", url)
        self.assertIn("priceFrom=500000", url)
        self.assertIn("priceTo=1000000", url)
        self.assertIn("propertyTypes=House", url)
        
        print(f"\nBasic params URL: {url}")
    
    def test_build_search_url_with_multiple_locations(self):
        # Test with multiple locations
        search_params = PropertySearchRequest(
            location=["NSW-Bondi-2026", "NSW-Chatswood-2067"],
            min_price=500000,
            max_price=1000000
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("loc=nsw-bondi-2026%2Cnsw-chatswood-2067", url)
        
        print(f"\nMultiple locations URL: {url}")
    
    def test_build_search_url_with_multiple_property_types(self):
        # Test with multiple property types
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"],
            property_type=["house", "apartment", "townhouse"]
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("propertyTypes=House%2CApartment%2CTownhouse", url)
        self.assertIn("loc=nsw-sydney-2000", url)
        
        print(f"\nMultiple property types URL: {url}")
    
    def test_build_search_url_with_all_params(self):
        # Test with all parameters
        search_params = PropertySearchRequest(
            location=["NSW-Bondi-2026", "NSW-Chatswood-2067"],
            min_price=500000,
            max_price=1000000,
            min_bedrooms=2,
            min_bathrooms=1,
            property_type=["house", "apartment"],
            car_parks=1,
            land_size_from=200,
            land_size_to=3000
        )
        
        url = self.scraper._build_search_url(search_params)
        
        # Check that the URL includes all parameters
        self.assertIn("2-bedrooms", url)
        self.assertIn("loc=nsw-bondi-2026%2Cnsw-chatswood-2067", url)
        self.assertIn("priceFrom=500000", url)
        self.assertIn("priceTo=1000000", url)
        self.assertIn("propertyTypes=House%2CApartment", url)
        self.assertIn("cars=1", url)
        self.assertIn("bathrooms=1", url)
        self.assertIn("landSizeFrom=200", url)
        self.assertIn("landSizeTo=3000", url)
        
        print(f"\nAll params URL: {url}")
    
    def test_build_search_url_with_land_size_from(self):
        # Test with land_size_from parameter
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"],
            land_size_from=200
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("landSizeFrom=200", url)
        self.assertIn("loc=nsw-sydney-2000", url)
        
        print(f"\nLand size from URL: {url}")

    def test_build_search_url_matching_example(self):
        """Test that we can generate a URL similar to the example:
        https://view.com.au/for-sale/2-bedrooms/?bathrooms=1&cars=1&loc=nsw-chatsworth-2469%2Cnsw-bondi-2026&priceFrom=200000&priceTo=2500000&landSizeFrom=200&landSizeTo=3000&propertyTypes=Apartment%2CHouse%2CUnit%2CTownhouse%2CVilla%2CRural
        """
        search_params = PropertySearchRequest(
            location=["nsw-chatsworth-2469", "nsw-bondi-2026"],
            min_price=200000,
            max_price=2500000,
            min_bedrooms=2,
            min_bathrooms=1,
            car_parks=1,
            property_type=["Apartment", "House", "Unit", "Townhouse", "Villa", "Rural"],
            land_size_from=200,
            land_size_to=3000
        )
        
        url = self.scraper._build_search_url(search_params)
        
        # Verify all components of the URL are present
        self.assertIn("for-sale/2-bedrooms/", url)
        self.assertIn("bathrooms=1", url)
        self.assertIn("cars=1", url)
        self.assertIn("loc=nsw-chatsworth-2469%2Cnsw-bondi-2026", url)
        self.assertIn("priceFrom=200000", url)
        self.assertIn("priceTo=2500000", url)
        self.assertIn("landSizeFrom=200", url)
        self.assertIn("landSizeTo=3000", url)
        self.assertIn("propertyTypes=Apartment%2CHouse%2CUnit%2CTownhouse%2CVilla%2CRural", url)
        
        print(f"\nExample match URL: {url}")
        print("Expected URL format: https://view.com.au/for-sale/2-bedrooms/?bathrooms=1&cars=1&loc=nsw-chatsworth-2469%2Cnsw-bondi-2026&priceFrom=200000&priceTo=2500000&landSizeFrom=200&landSizeTo=3000&propertyTypes=Apartment%2CHouse%2CUnit%2CTownhouse%2CVilla%2CRural")

    # 新增更全面的测试用例
    def test_build_search_url_with_only_location(self):
        # 只有位置参数的情况
        search_params = PropertySearchRequest(
            location=["VIC-Melbourne-3000"]
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("loc=vic-melbourne-3000", url)
        self.assertNotIn("priceFrom", url)
        self.assertNotIn("priceTo", url)
        
        print(f"\nOnly location URL: {url}")
    
    def test_build_search_url_with_only_price_range(self):
        # 只有价格范围的情况（加上必须的location参数）
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"],
            min_price=300000,
            max_price=800000
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("loc=nsw-sydney-2000", url)
        self.assertIn("priceFrom=300000", url)
        self.assertIn("priceTo=800000", url)
        
        print(f"\nOnly price range URL: {url}")
    
    def test_build_search_url_with_only_min_price(self):
        # 只有最低价格的情况（加上必须的location参数）
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"],
            min_price=500000
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("loc=nsw-sydney-2000", url)
        self.assertIn("priceFrom=500000", url)
        self.assertNotIn("priceTo", url)
        
        print(f"\nOnly min price URL: {url}")
    
    def test_build_search_url_with_only_max_price(self):
        # 只有最高价格的情况（加上必须的location参数）
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"],
            max_price=1500000
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("loc=nsw-sydney-2000", url)
        self.assertIn("priceTo=1500000", url)
        self.assertNotIn("priceFrom", url)
        
        print(f"\nOnly max price URL: {url}")
    
    def test_build_search_url_with_only_bedrooms(self):
        # 只有卧室数量的情况（加上必须的location参数）
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"],
            min_bedrooms=3
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("3-bedrooms", url)
        self.assertIn("loc=nsw-sydney-2000", url)
        
        print(f"\nOnly bedrooms URL: {url}")
    
    def test_build_search_url_with_only_bathrooms(self):
        # 只有浴室数量的情况（加上必须的location参数）
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"],
            min_bathrooms=2
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("bathrooms=2", url)
        self.assertIn("loc=nsw-sydney-2000", url)
        
        print(f"\nOnly bathrooms URL: {url}")
    
    def test_build_search_url_with_only_property_type(self):
        # 只有房产类型的情况（加上必须的location参数）
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"],
            property_type=["townhouse"]
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("propertyTypes=Townhouse", url)
        self.assertIn("loc=nsw-sydney-2000", url)
        
        print(f"\nOnly property type URL: {url}")
    
    def test_build_search_url_with_only_car_parks(self):
        # 只有停车位的情况（加上必须的location参数）
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"],
            car_parks=2
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("cars=2", url)
        self.assertIn("loc=nsw-sydney-2000", url)
        
        print(f"\nOnly car parks URL: {url}")
    
    def test_build_search_url_with_only_land_size(self):
        # 只有土地面积的情况（加上必须的location参数）
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"],
            land_size_from=500,
            land_size_to=1000
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("landSizeFrom=500", url)
        self.assertIn("landSizeTo=1000", url)
        self.assertIn("loc=nsw-sydney-2000", url)
        
        print(f"\nOnly land size URL: {url}")

    def test_build_search_url_with_only_land_size_from(self):
        # 只有最小土地面积的情况（加上必须的location参数）
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"],
            land_size_from=600
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("landSizeFrom=600", url)
        self.assertNotIn("landSizeTo", url)
        self.assertIn("loc=nsw-sydney-2000", url)
        
        print(f"\nOnly land size from URL: {url}")
    
    def test_build_search_url_with_only_land_size_to(self):
        # 只有最大土地面积的情况（加上必须的location参数）
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"],
            land_size_to=800
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("landSizeTo=800", url)
        self.assertNotIn("landSizeFrom", url)
        self.assertIn("loc=nsw-sydney-2000", url)
        
        print(f"\nOnly land size to URL: {url}")
    
    def test_build_search_url_with_many_locations(self):
        # 测试多个地点（超过2个）
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000", "VIC-Melbourne-3000", "QLD-Brisbane-4000", "SA-Adelaide-5000"]
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("loc=nsw-sydney-2000%2Cvic-melbourne-3000%2Cqld-brisbane-4000%2Csa-adelaide-5000", url)
        
        print(f"\nMany locations URL: {url}")
    
    def test_build_search_url_with_many_property_types(self):
        # 测试多个房产类型（加上必须的location参数）
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"],
            property_type=["house", "apartment", "unit", "townhouse", "villa", "rural", "land"]
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("propertyTypes=House%2CApartment%2CUnit%2CTownhouse%2CVilla%2CRural%2CLand", url)
        self.assertIn("loc=nsw-sydney-2000", url)
        
        print(f"\nMany property types URL: {url}")
    
    def test_build_search_url_with_extreme_values(self):
        # 测试极端值（加上必须的location参数）
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"],
            min_price=1,
            max_price=100000000,  # 1亿
            min_bedrooms=10,
            min_bathrooms=8,
            car_parks=10,
            land_size_from=1,
            land_size_to=100000  # 10万平方米
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("10-bedrooms", url)
        self.assertIn("loc=nsw-sydney-2000", url)
        self.assertIn("priceFrom=1", url)
        self.assertIn("priceTo=100000000", url)
        self.assertIn("bathrooms=8", url)
        self.assertIn("cars=10", url)
        self.assertIn("landSizeFrom=1", url)
        self.assertIn("landSizeTo=100000", url)
        
        print(f"\nExtreme values URL: {url}")
    
    def test_build_search_url_complex_combination(self):
        # 复杂组合1：高价公寓，要有停车位，不需要太大土地
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000", "NSW-North-Sydney-2060"],
            min_price=2000000,
            max_price=5000000,
            min_bedrooms=3,
            min_bathrooms=2,
            property_type=["apartment", "unit"],
            car_parks=2,
            land_size_from=None,
            land_size_to=None
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("3-bedrooms", url)
        self.assertIn("loc=nsw-sydney-2000%2Cnsw-north-sydney-2060", url)
        self.assertIn("priceFrom=2000000", url)
        self.assertIn("priceTo=5000000", url)
        self.assertIn("propertyTypes=Apartment%2CUnit", url)
        self.assertIn("bathrooms=2", url)
        self.assertIn("cars=2", url)
        self.assertNotIn("landSizeFrom", url)
        self.assertNotIn("landSizeTo", url)
        
        print(f"\nComplex combination 1 URL: {url}")
    
    def test_build_search_url_complex_combination_2(self):
        # 复杂组合2：家庭住宅，要有大土地，不限价格
        search_params = PropertySearchRequest(
            location=["NSW-Hornsby-2077", "NSW-Castle-Hill-2154"],
            min_price=None,
            max_price=None,
            min_bedrooms=4,
            min_bathrooms=2,
            property_type=["house"],
            car_parks=2,
            land_size_from=600,
            land_size_to=None
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("4-bedrooms", url)
        self.assertIn("loc=nsw-hornsby-2077%2Cnsw-castle-hill-2154", url)
        self.assertNotIn("priceFrom", url)
        self.assertNotIn("priceTo", url)
        self.assertIn("propertyTypes=House", url)
        self.assertIn("bathrooms=2", url)
        self.assertIn("cars=2", url)
        self.assertIn("landSizeFrom=600", url)
        self.assertNotIn("landSizeTo", url)
        
        print(f"\nComplex combination 2 URL: {url}")
    
    def test_build_search_url_complex_combination_3(self):
        # 复杂组合3：投资型房产，各种类型都可以考虑
        search_params = PropertySearchRequest(
            location=["QLD-Brisbane-4000", "QLD-Gold-Coast-4217"],
            min_price=400000,
            max_price=800000,
            min_bedrooms=2,
            min_bathrooms=1,
            property_type=["house", "apartment", "townhouse", "unit"],
            car_parks=1,
            land_size_from=None,
            land_size_to=None
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertIn("2-bedrooms", url)
        self.assertIn("loc=qld-brisbane-4000%2Cqld-gold-coast-4217", url)
        self.assertIn("priceFrom=400000", url)
        self.assertIn("priceTo=800000", url)
        self.assertIn("propertyTypes=House%2CApartment%2CTownhouse%2CUnit", url)
        self.assertIn("bathrooms=1", url)
        self.assertIn("cars=1", url)
        
        print(f"\nComplex combination 3 URL: {url}")

    def test_build_search_url_minimal_params(self):
        # 测试最小参数情况（至少需要一个位置）
        search_params = PropertySearchRequest(
            location=["NSW-Sydney-2000"]
        )
        
        url = self.scraper._build_search_url(search_params)
        self.assertEqual(url, "https://view.com.au/for-sale/?loc=nsw-sydney-2000")
        
        print(f"\nMinimal params URL: {url}")

if __name__ == "__main__":
    unittest.main() 