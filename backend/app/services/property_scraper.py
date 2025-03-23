import requests
from bs4 import BeautifulSoup
import json
import time
import random
from typing import List, Dict, Optional
from fake_useragent import UserAgent
import logging
import re
from app.models import PropertySearchRequest, PropertySearchResponse

logging.basicConfig(level=logging.INFO)

class PropertyScraper:
    def __init__(self):
        self.base_url = "https://www.view.com.au"
        self.headers = {
            'User-Agent': UserAgent().random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }

    def _build_search_url(self, search_params: PropertySearchRequest) -> str:
        """Build the search URL with the given filters"""
        # 使用不带www的域名
        base_url = "https://view.com.au"
        search_url = f"{base_url}/for-sale/"
        
        # Add bedrooms filter if specified
        if search_params.get("min_bedrooms"):
            search_url += f"{search_params['min_bedrooms']}-bedrooms/"
            
        # Start with query parameters
        search_url += "?"
        
        # 参数按照示例URL的顺序添加
        # 1. 先添加bathrooms（如果有）
        if search_params.get("min_bathrooms"):
            search_url += f"bathrooms={search_params['min_bathrooms']}"
            # 添加后面的参数需要&前缀
            have_params = True
        else:
            have_params = False
            
        # 2. 添加cars（如果有）
        if search_params.get("car_parks"):
            if have_params:
                search_url += "&"
            search_url += f"cars={search_params['car_parks']}"
            have_params = True
            
        # 3. 添加location（如果有）
        locations = search_params.get("location")
        if locations and isinstance(locations, list) and len(locations) > 0:
            if have_params:
                search_url += "&"
            # 格式化locations并进行URL编码（将逗号编码为%2C）
            formatted_locations = [loc.lower().replace(' ', '-') for loc in locations]
            search_url += f"loc={','.join(formatted_locations).replace(',', '%2C')}"
            have_params = True
        
        # 4. 添加价格范围
        if search_params.get("min_price"):
            if have_params:
                search_url += "&"
            search_url += f"priceFrom={int(search_params['min_price'])}"
            have_params = True
            
        if search_params.get("max_price"):
            if have_params:
                search_url += "&"
            search_url += f"priceTo={int(search_params['max_price'])}"
            have_params = True
            
        # 5. 添加土地面积
        if search_params.get("land_size_from"):
            if have_params:
                search_url += "&"
            search_url += f"landSizeFrom={int(search_params['land_size_from'])}"
            have_params = True
            
        if search_params.get("land_size_to"):
            if have_params:
                search_url += "&"
            search_url += f"landSizeTo={int(search_params['land_size_to'])}"
            have_params = True
            
        # 6. 最后添加property_type
        property_types = search_params.get("property_type")
        if property_types and isinstance(property_types, list) and len(property_types) > 0:
            if have_params:
                search_url += "&"
            # 格式化property_types并进行URL编码（将逗号编码为%2C）
            formatted_types = [p_type.capitalize() for p_type in property_types]
            search_url += f"propertyTypes={','.join(formatted_types).replace(',', '%2C')}"
            
        return search_url
        
    def _get_page(self, url: str) -> Optional[str]:
        """Make HTTP request with error handling"""
        try:
            # Random delay between requests (1-3 seconds)
            time.sleep(random.uniform(1, 3))
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            return None

    def _parse_listing(self, listing) -> Optional[PropertySearchResponse]:
        """Parse individual listing element and return PropertySearchResponse"""
        try:
            # Extract all required fields
            listing_id = listing.get('data-testid')
            
            price = (listing.find('p', {'data-testid': 'property-card-title'}).get_text(strip=True) 
                    if listing.find('p', {'data-testid': 'property-card-title'}) 
                    else 'Contact agent')
            
            address = (listing.find('h2').get_text(strip=True)
                      if listing.find('h2')
                      else 'No address')
            
            bedrooms = (listing.find('div', {'data-testid': re.compile(".*bedroom*")}).find('span').get_text(strip=True)
                       if listing.find('div', {'data-testid': re.compile(".*bedroom*")})
                       else '0')
            
            bathrooms = (listing.find('div', {'data-testid': re.compile(".*bathroom*")}).find('span').get_text(strip=True)
                        if listing.find('div', {'data-testid': re.compile(".*bathroom*")})
                        else '0')
            
            car_parks = (listing.find('div', {'data-testid': re.compile(".*carpark*")}).find('span').get_text(strip=True)
                        if listing.find('div', {'data-testid': re.compile(".*carpark*")})
                        else None)
            
            land_size = (listing.find('div', {'data-testid': re.compile(".*land-size*")}).find('span').get_text(strip=True)
                        if listing.find('div', {'data-testid': re.compile(".*land-size*")})
                        else None)
            
            property_type = (listing.find('span', class_=re.compile(r'(?=.*\btext-xs\b)(?=.*\bw-fit\b)')).get_text(strip=True)
                           if listing.find('span', class_=re.compile(r'(?=.*\btext-xs\b)(?=.*\bw-fit\b)'))
                           else None)
            
            inspection_date = (listing.find('h4', {'data-testid': 'date-text-tag'}).get_text(strip=True)
                             if listing.find('h4', {'data-testid': 'date-text-tag'})
                             else None)
            
            image_urls = ([img['src'] for img in listing.find_all('img', {'class': 'image-gallery-image'})]
                        if listing.find_all('img', {'class': 'image-gallery-image'})
                        else None)
            
            agent_name = (listing.find('div', {'data-testid': 'agency-image'}).find('img').get('alt')
                         if listing.find('div', {'data-testid': 'agency-image'})
                         else None)
            
            # Create and return PropertySearchResponse object
            return PropertySearchResponse(
                listing_id=listing_id,
                price=price,
                address=address,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                car_parks=car_parks,
                land_size=land_size,
                property_type=property_type,
                inspection_date=inspection_date,
                image_urls=image_urls,
                agent_name=agent_name
            )
            
        except Exception as e:
            logging.error(f"Error parsing listing: {str(e)}")
            return None

    async def search_properties(
        self,
        search_params: PropertySearchRequest,
        max_results: int = 20
    ) -> List[PropertySearchResponse]:
        """
        Search properties with given criteria
        Returns list of PropertySearchResponse objects matching the criteria
        """
        try:
            # Construct search URL with filters
            search_url = self._build_search_url(search_params)
            logging.info(f"Searching properties with URL: {search_url}")
            
            # Log search parameters for debugging
            logging.debug(f"Search parameters: {search_params}")
            
            # Get search results page
            html = self._get_page(search_url)
            if not html:
                logging.error("Failed to fetch search results page")
                return []

            # Parse the page
            soup = BeautifulSoup(html, 'html.parser')
            pattern = re.compile(r'listing-\d+')
            listing_elements = soup.find_all('span', {'data-testid': pattern})
            
            logging.info(f"Found {len(listing_elements)} raw listings on page")

            # Parse and filter listings
            results = []
            for listing in listing_elements:
                if len(results) >= max_results:
                    break

                property_response = self._parse_listing(listing)
                if property_response:
                    results.append(property_response)

            logging.info(f"Successfully parsed {len(results)} properties matching criteria")
            return results

        except Exception as e:
            logging.error(f"Error in search_properties: {str(e)}")
            return []

    def extract_price(self, price_str: str) -> Optional[float]:
        """Extract numeric price from string"""
        try:
            # Remove currency symbol and commas
            price_str = price_str.replace('$', '').replace(',', '')
            # Convert to float
            return float(price_str)
        except (ValueError, AttributeError):
            return None 
        
