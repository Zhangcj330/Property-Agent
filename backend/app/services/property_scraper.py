import requests
from bs4 import BeautifulSoup
import json
import time
import random
from typing import List, Dict, Optional
from fake_useragent import UserAgent
import logging

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

    def _build_search_url(
        self,
        location: str,
        min_beds: Optional[int] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        property_type: Optional[str] = None
    ) -> str:
        """Build the search URL with the given filters"""
        search_url = f"{self.base_url}/for-sale/"
        
        # Add bedrooms filter if specified
        if min_beds:
            search_url += f"{min_beds}-bedrooms/"
            
        # Add location filter
        search_url += f"?loc={location.lower().replace(' ', '-')}"
        
        # Add price range filters
        if min_price:
            search_url += f"&priceFrom={int(min_price)}"
        if max_price:
            search_url += f"&priceTo={int(max_price)}"
            
        # Add property type filter
        if property_type:
            search_url += f"&propertyTypes={property_type.capitalize()}"
            
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

    def _parse_listing(self, listing) -> Optional[Dict]:
        """Parse individual listing element"""
        try:
            return {
                'price': (listing.find('p', {'data-testid': 'property-card-title'}).get_text(strip=True) 
                         if listing.find('p', {'data-testid': 'property-card-title'}) 
                         else None),
                
                'address': (listing.find('h2').get_text(strip=True)
                           if listing.find('h2')
                           else None),
                
                'bedrooms': int((listing.find('div', {'data-testid': 'a-bedrooms'}).find('span').get_text(strip=True)
                            if listing.find('div', {'data-testid': 'a-bedrooms'})
                            else '0')),
                
                'bathrooms': int((listing.find('div', {'data-testid': 'a-bathrooms'}).find('span').get_text(strip=True)
                             if listing.find('div', {'data-testid': 'a-bathrooms'})
                             else '0')),
                
                'car_parks': int((listing.find('div', {'data-testid': 'a-carparks'}).find('span').get_text(strip=True)
                             if listing.find('div', {'data-testid': 'a-carparks'})
                             else '0')),
                
                'land_size': (listing.find('div', {'data-testid': 'a-land-size'}).find('span').get_text(strip=True)
                             if listing.find('div', {'data-testid': 'a-land-size'})
                             else None),
                
                'property_type': (listing.find('span', class_='text-xs').get_text(strip=True).lower()
                                if listing.find('span', class_='text-xs')
                                else None),
                
                'image_url': (listing.find('img', {'class': 'image-gallery-image'})['src']
                             if listing.find('img', {'class': 'image-gallery-image'})
                             else None),
            }
        except Exception as e:
            logging.error(f"Error parsing listing: {str(e)}")
            return None

    async def search_properties(
        self,
        location: str,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_beds: Optional[int] = None,
        property_type: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search properties with given criteria
        Returns list of properties matching the criteria
        """
        try:
            # Construct search URL with filters
            search_url = self._build_search_url(
                location=location,
                min_beds=min_beds,
                min_price=min_price,
                max_price=max_price,
                property_type=property_type
            )
            print(search_url)
            # Get search results page
            html = self._get_page(search_url)
            if not html:
                return []

            # Parse the page
            soup = BeautifulSoup(html, 'html.parser')
            listing_elements = soup.find_all(
                'div', 
                class_="relative flex flex-col bg-at-white rounded-none md-744:rounded-xl overflow-hidden cursor-pointer"
            )

            # Parse and filter listings
            results = []
            for listing in listing_elements:
                if len(results) >= max_results:
                    break
                print(listing)
                print(111111)
                listing_data = self._parse_listing(listing)
                if not listing_data:
                    continue
                print(listing_data)
                print(222222)
                # Apply filters
                
                results.append(listing_data)

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