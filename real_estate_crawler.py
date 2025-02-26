import requests
from bs4 import BeautifulSoup
import json
import time
import random
from urllib.parse import urljoin
import pandas as pd
from fake_useragent import UserAgent
import logging
import csv
import os
from ec2_manager import EC2Manager

class ViewComCrawler:
    def __init__(self):
        self.base_url = "https://www.view.com.au"
        self.headers = {
            'User-Agent': UserAgent().random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        self.listings = []
        self.start_time = None
        self.timeout_minutes = 60  # Stop after 60 minutes
        self.ec2_manager = EC2Manager()
        self.max_retries = 3
        self.current_retry = 0

    def get_page(self, url):
        """Make HTTP request with error handling and IP rotation"""
        try:
            # Random delay between requests (1-3 seconds)
            time.sleep(random.uniform(1, 3))
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logging.warning("Received 403 Forbidden - IP might be blocked")
                if self.current_retry < self.max_retries:
                    self.current_retry += 1
                    logging.info(f"Attempting to switch to new EC2 instance (Attempt {self.current_retry}/{self.max_retries})")
                    
                    # Terminate current instance and create new one
                    self.ec2_manager.terminate_instance()
                    new_ip = self.ec2_manager.create_instance()
                    
                    if new_ip:
                        logging.info(f"Successfully switched to new IP: {new_ip}")
                        # Wait for instance to fully initialize
                        time.sleep(60)
                        # Retry the request
                        return self.get_page(url)
                    else:
                        logging.error("Failed to create new EC2 instance")
                else:
                    logging.error("Max retries reached for creating new instances")
            
            logging.error(f"Error fetching {url}: {str(e)}")
            return None
            
        except requests.RequestException as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            return None

    def parse_listing(self, listing):
        """Parse individual listing element"""
        try:
            listing_data = {
                # Price
                'price': (listing.find('p', {'data-testid': 'property-card-title'}).get_text(strip=True) 
                         if listing.find('p', {'data-testid': 'property-card-title'}) 
                         else 'No price'),
                
                # Address
                'address': (listing.find('h2').get_text(strip=True)
                           if listing.find('h2')
                           else 'No address'),
                
                # Property features
                'bedrooms': (listing.find('div', {'data-testid': 'a-bedrooms'}).find('span').get_text(strip=True)
                            if listing.find('div', {'data-testid': 'a-bedrooms'})
                            else '0'),
                
                'bathrooms': (listing.find('div', {'data-testid': 'a-bathrooms'}).find('span').get_text(strip=True)
                             if listing.find('div', {'data-testid': 'a-bathrooms'})
                             else '0'),
                
                'car_parks': (listing.find('div', {'data-testid': 'a-carparks'}).find('span').get_text(strip=True)
                             if listing.find('div', {'data-testid': 'a-carparks'})
                             else '0'),
                
                # Land size
                'land_size': (listing.find('div', {'data-testid': 'a-land-size'}).find('span').get_text(strip=True)
                             if listing.find('div', {'data-testid': 'a-land-size'})
                             else 'No land size'),
                
                # Property type
                'property_type': (listing.find('span', class_='text-xs').get_text(strip=True)
                                if listing.find('span', class_='text-xs')
                                else 'No property type'),
                
                # Inspection date
                'inspection_date': (listing.find('h4', {'data-testid': 'date-text-tag'}).get_text(strip=True)
                                  if listing.find('h4', {'data-testid': 'date-text-tag'})
                                  else 'No inspection date'),
                
                # Image URL
                'image_url': (listing.find('img', {'class': 'image-gallery-image'})['src']
                             if listing.find('img', {'class': 'image-gallery-image'})
                             else 'No image available'),
                
                # Agent name
                'agent_name': 'No agent name'  # Default value
            }
            
            # Special handling for agent name since it requires nested checks
            agent_image_div = listing.find('div', {'data-testid': 'agency-image'})
            if agent_image_div and agent_image_div.find('img'):
                listing_data['agent_name'] = agent_image_div.find('img').get('alt', 'No agent name')
            
            return listing_data
            
        except Exception as e:
            logging.error(f"Error parsing listing: {str(e)}")
            return None

    def initialize_csv(self, filename='listings.csv'):
        """Initialize CSV file with headers if it doesn't exist"""
        headers = [
            'price', 'address', 'bedrooms', 'bathrooms', 'car_parks',
            'land_size', 'property_type', 'inspection_date', 'image_url',
            'agent_name'
        ]
        
        # Check if file exists
        if not os.path.exists(filename):
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                logging.info(f"Created new CSV file: {filename}")
        else:
            logging.info(f"Using existing CSV file: {filename}")

    def append_to_csv(self, listing_data, filename='listings.csv', check_duplicates=False):
        """
        Append a single listing to CSV file
        :param listing_data: Dictionary containing listing information
        :param filename: Path to CSV file
        :param check_duplicates: Whether to check for duplicates (may be slow for large files)
        """
        try:
            # Check for duplicates if requested
            if check_duplicates:
                try:
                    df = pd.read_csv(filename)
                    # Check if listing with same address exists
                    if not df[df['address'] == listing_data['address']].empty:
                        logging.info(f"Duplicate listing found for {listing_data['address']}. Skipping...")
                        return False
                except Exception as e:
                    logging.error(f"Error checking duplicates: {str(e)}")
                    # Continue with append if duplicate check fails
                    pass

            # Append the new listing
            with open(filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=listing_data.keys())
                writer.writerow(listing_data)
            return True
            
        except Exception as e:
            logging.error(f"Error writing to CSV: {str(e)}")
            return False

    def crawl_search_results(self, search_url, max_pages=None, filename='listings.csv', check_duplicates=False):
        """
        Crawl through search results pages and save incrementally
        :param search_url: URL to crawl
        :param max_pages: Maximum number of pages to crawl
        :param filename: Output CSV file
        :param check_duplicates: Whether to check for duplicates before adding new listings
        """
        self.start_time = time.time()
        page = 1
        total_listings = 0
        
        # Initialize or use existing CSV file
        self.initialize_csv(filename)
        
        while True:
            # Timeout check
            if time.time() - self.start_time > self.timeout_minutes * 60:
                logging.info("Crawler timeout reached. Stopping...")
                break
                
            
            # Max pages check
            if max_pages and page > max_pages:
                logging.info(f"Reached maximum number of pages ({max_pages}). Stopping...")
                break

            try:
                current_url = f"{search_url}?page={page}"
                html = self.get_page(current_url)
                
                if not html:
                    logging.info("No more pages to crawl. Stopping...")
                    break

                soup = BeautifulSoup(html, 'html.parser')
                listing_elements = soup.find_all('div', class_="relative flex flex-col bg-at-white rounded-none md-744:rounded-xl overflow-hidden cursor-pointer")
                
                if not listing_elements:
                    logging.info("No more listings found. Stopping...")
                    break

                # Process each listing and write to CSV immediately
                for listing in listing_elements:
                    listing_data = self.parse_listing(listing)
                    
                    if listing_data:
                        # Write to CSV immediately instead of storing in memory
                        if self.append_to_csv(listing_data, filename, check_duplicates):
                            total_listings += 1
                            logging.info(f"Successfully scraped and saved listing: {listing_data['address']} (Total: {total_listings})")
                        

                logging.info(f"Completed page {page} (Total listings: {total_listings})")
                page += 1
                
            except Exception as e:
                logging.error(f"Error crawling page {page}: {str(e)}")
                break
        
        return total_listings

def main():
    crawler = ViewComCrawler()
    
    # Example search URL - modify as needed
    search_url = "https://www.view.com.au/for-sale/nsw/sydney"
    
    # Crawl and save incrementally
    total_listings = crawler.crawl_search_results(
        search_url, 
        max_pages=2,
        filename='sydney_listings.csv'
    )
    
    logging.info(f"Crawling completed. Total listings saved: {total_listings}")

if __name__ == "__main__":
    main() 