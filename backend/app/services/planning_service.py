import pandas as pd
import time
import asyncio
import aiohttp
from typing import Optional, Dict, Any, Tuple
import backoff
from aiohttp import ClientSession, ClientResponseError
from asyncio import Semaphore
import json
from pathlib import Path
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import os
import logging
import aiofiles

logger = logging.getLogger(__name__)

# Rate limiting settings
MAX_CONCURRENT_REQUESTS = 100
REQUEST_DELAY = 0.05

# Cache settings
CACHE_DIR = Path("cache")
CACHE_DURATION = timedelta(days=7)  # Cache results for 7 days

# Layer definitions
PLANNING_LAYERS = {
    'heritage': 0,           # Heritage
    'fsr': 1,               # Floor Space Ratio
    'zoning': 2,            # Land Zoning
    'lot_size': 4,          # Lot Size
    'height': 5,            # Height of Building
}

HAZARD_LAYERS = {
    'flood': 1,      # Flood Planning
    'landslide': 2   # Landslide Risk Land
}

# Cache coordinates for addresses to avoid repeated geocoding
address_cache = {}

# Custom exception for rate limiting
class RateLimitError(Exception):
    """Exception raised when API enforces rate limiting"""
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Rate limited, retry after {retry_after} seconds")

class Location(BaseModel):
    address: str
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)

class PlanningInfo(BaseModel):
    location: Location
    # Planning data
    zone_name: Optional[str] = None
    height_limit: Optional[str] = None
    floor_space_ratio: Optional[float] = None
    min_lot_size: Optional[str] = None
    is_heritage: bool = False
    # Hazard data
    flood_risk: bool = False
    landslide_risk: bool = False
    # Metadata
    timestamp: float = Field(default_factory=time.time)
    error: Optional[str] = None
    source: str = "fresh"

    @property
    def has_coordinates(self) -> bool:
        return self.location.latitude is not None and self.location.longitude is not None

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of key planning information"""
        return {
            "address": self.location.address,
            "has_coordinates": self.has_coordinates,
            "zone_name": self.zone_name,
            "height_limit": self.height_limit,
            "floor_space_ratio": self.floor_space_ratio,
            "min_lot_size": self.min_lot_size,
            "is_heritage": self.is_heritage,
            "flood_risk": self.flood_risk,
            "landslide_risk": self.landslide_risk
        }

def get_cache_key(address: str) -> str:
    """Generate a cache key from an address"""
    return address.lower().replace(" ", "_").replace("/", "_")

async def get_cached_result(address: str) -> Optional[Dict[str, Any]]:
    """Try to get cached result for an address asynchronously"""
    try:
        cache_key = get_cache_key(address)
        cache_file = CACHE_DIR / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        async with aiofiles.open(cache_file, 'r') as f:
            content = await f.read()
            data = json.loads(content)
            
        # Check if cache is still valid
        timestamp = data.get('timestamp', 0)
        if time.time() - timestamp > CACHE_DURATION.total_seconds():
            return None
            
        return data
        
    except Exception as e:
        logger.error(f"Cache read error: {e}")
        return None

def clean_address(address: str) -> str:
    """Clean address by removing everything before and including the '/' character"""
    address = address.lower().strip()
    if '/' in address:
        address = address.split('/', 1)[1]
    return address.strip()

async def get_coordinates(address: str) -> Tuple[Optional[float], Optional[float]]:
    """Get coordinates for an address using Google Maps API (async)"""
    # Check cache first
    if address in address_cache:
        return address_cache[address]
        
    try:
        api_key = os.environ.get('GOOGLE_MAP_API')
        if not api_key:
            logger.error("GOOGLE_MAPS_API_KEY not found in environment variables")
            return None, None
            
        address_cleaned = clean_address(address)
        url = f"https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": address_cleaned,
            "key": api_key
        }
        
        logger.debug(f"Requesting coordinates for address: {address_cleaned}")
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Geocoding error: HTTP {response.status}")
                    return None, None
                    
                data = await response.json()
        
        elapsed = time.time() - start_time
        logger.debug(f"Google geocoding API response received in {elapsed:.2f}s")
        
        if data['status'] != 'OK':
            logger.error(f"Geocoding error: {data['status']}")
            return None, None
            
        if not data['results']:
            logger.warning(f"No geocoding results for address: {address}")
            return None, None
            
        location = data['results'][0]['geometry']['location']
        lat, lng = location['lat'], location['lng']
        
        # Cache the result
        address_cache[address] = (lat, lng)
        logger.info(f"Coordinates found for {address}")
        
        return lat, lng
        
    except Exception as e:
        logger.error(f"Error getting coordinates for {address}: {str(e)}")
        return None, None

@backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError, RateLimitError), max_tries=3)
async def query_planning_layers(session: ClientSession, layer_index: int, lon: float, lat: float, semaphore: Semaphore) -> Optional[Dict[str, Any]]:
    """Query the EPI Primary Planning Layers MapServer"""
    logger.info(f"[PLANNING] Layer {layer_index} query START at {time.time():.3f}")
    layer_start_time = time.time()
    async with semaphore:
        await asyncio.sleep(REQUEST_DELAY)
        base_url = "https://mapprod3.environment.nsw.gov.au/arcgis/rest/services/Planning/EPI_Primary_Planning_Layers/MapServer"
        query_url = f"{base_url}/{layer_index}/query"
        params = {
            "geometry": f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "inSR": "4283",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "json"
        }
        
        req_start = time.time()
        try:
            async with session.get(query_url, params=params, timeout=15) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', '60'))
                    raise RateLimitError(retry_after)
                
                data = await response.json()
                req_end = time.time()
                logger.info(f"[PLANNING] Layer {layer_index} query END at {time.time():.3f}, duration: {req_end - req_start:.3f}s")
                logger.debug(f"Planning layer {layer_index} query took {req_end - req_start:.2f}s")
                return data if 'features' in data else None
        except Exception as e:
            req_end = time.time()
            logger.info(f"[PLANNING] Layer {layer_index} query END (EXCEPTION) at {time.time():.3f}, duration: {req_end - req_start:.3f}s")
            logger.error(f"Planning layer {layer_index} query failed after {req_end - req_start:.2f}s: {str(e)}")
            raise

@backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError, RateLimitError), max_tries=3)
async def query_hazard_layers(session: ClientSession, layer_index: int, lon: float, lat: float, semaphore: Semaphore) -> Optional[Dict[str, Any]]:
    """Query the Hazard MapServer"""
    logger.info(f"[HAZARD] Layer {layer_index} query START at {time.time():.3f}")
    layer_start_time = time.time()
    async with semaphore:
        await asyncio.sleep(REQUEST_DELAY)
        base_url = "https://mapprod3.environment.nsw.gov.au/arcgis/rest/services/Planning/Hazard/MapServer"
        query_url = f"{base_url}/{layer_index}/query"
        params = {
            "geometry": f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "inSR": "4283",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "json"
        }
        
        req_start = time.time()
        try:
            async with session.get(query_url, params=params, timeout=15) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', '60'))
                    raise RateLimitError(retry_after)
                
                data = await response.json()
                req_end = time.time()
                logger.info(f"[HAZARD] Layer {layer_index} query END at {time.time():.3f}, duration: {req_end - req_start:.3f}s")
                logger.debug(f"Hazard layer {layer_index} query took {req_end - req_start:.2f}s")
                return data if 'features' in data else None
        except Exception as e:
            req_end = time.time()
            logger.info(f"[HAZARD] Layer {layer_index} query END (EXCEPTION) at {time.time():.3f}, duration: {req_end - req_start:.3f}s")
            logger.error(f"Hazard layer {layer_index} query failed after {req_end - req_start:.2f}s: {str(e)}")
            raise

async def get_all_planning_info(address: str) -> PlanningInfo:
    """Get all planning information for a given address"""
    start_time = time.time()
    
    # Initialize with location
    location = Location(address=address)
    planning_info = PlanningInfo(location=location)
    
    try:
        # Get coordinates asynchronously
        logger.info(f"Getting coordinates for {address}")
        coordinates_start = time.time()
        lat, lon = await get_coordinates(address)
        coordinates_end = time.time()
        logger.info(f"Coordinates fetching took {coordinates_end - coordinates_start:.2f}s for {address}")
        
        if lat is None or lon is None:
            planning_info.error = f"Could not get coordinates for address: {address}"
            return planning_info
            
        logger.debug(f"Found coordinates: {lat}, {lon}")
        location.latitude = lat
        location.longitude = lon
        
        # Setup API client
        semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
        timeout = aiohttp.ClientTimeout(total=120)
        
        # Record start time for all queries
        all_queries_start = time.time()
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Prepare all query tasks
            planning_tasks = {}
            hazard_tasks = {}
            
            # Create all planning layer query tasks
            for layer_name, layer_index in PLANNING_LAYERS.items():
                logger.debug(f"Creating task for {layer_name} layer")
                planning_tasks[layer_name] = asyncio.create_task(
                    query_planning_layers(session, layer_index, lon, lat, semaphore)
                )
            
            # Create all hazard area query tasks
            for layer_name, layer_index in HAZARD_LAYERS.items():
                logger.debug(f"Creating task for {layer_name} hazard layer")
                hazard_tasks[layer_name] = asyncio.create_task(
                    query_hazard_layers(session, layer_index, lon, lat, semaphore)
                )
            
            # Wait for all queries to complete at once (more efficient than processing separately)
            all_tasks = list(planning_tasks.values()) + list(hazard_tasks.values())
            
            try:
                await asyncio.gather(*all_tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error in gather for planning/hazard queries: {e}")
                
            all_queries_end = time.time()
            logger.info(f"All planning/hazard queries took {all_queries_end - all_queries_start:.2f}s for {address}")
            
            # Process planning results
            logger.info("Processing planning results")
            for layer_name, task in planning_tasks.items():
                try:
                    if task.done():
                        data = task.result()
                        if isinstance(data, Exception):
                            logger.error(f"Error in {layer_name} task: {data}")
                            continue
                            
                        if data and data.get('features'):
                            attrs = data['features'][0]['attributes']
                            logger.debug(f"Processing {layer_name} attributes")
                            
                            if layer_name == 'zoning':
                                planning_info.zone_name = attrs.get('LAY_CLASS')
                            elif layer_name == 'height':
                                max_b_h = attrs.get('MAX_B_H')
                                units = attrs.get('UNITS', '')
                                if max_b_h is not None:
                                    if isinstance(max_b_h, float) and max_b_h.is_integer():
                                        max_b_h_str = str(int(max_b_h))
                                    else:
                                        max_b_h_str = str(max_b_h)
                                    planning_info.height_limit = f"{max_b_h_str} {units}".strip()
                                else:
                                    planning_info.height_limit = None
                            elif layer_name == 'fsr':
                                planning_info.floor_space_ratio = attrs.get('FSR')
                            elif layer_name == 'lot_size':
                                lot_size = attrs.get('LOT_SIZE')
                                units = attrs.get('UNITS', '')
                                if lot_size is not None:
                                    # 格式化 float，去掉无意义的小数点
                                    if isinstance(lot_size, float) and lot_size.is_integer():
                                        lot_size_str = str(int(lot_size))
                                    else:
                                        lot_size_str = str(lot_size)
                                    planning_info.min_lot_size = f"{lot_size_str} {units}".strip()
                                else:
                                    planning_info.min_lot_size = None
                            elif layer_name == 'heritage':
                                planning_info.is_heritage = True
                        else:
                            logger.debug(f"No {layer_name} data found")
                    else:
                        logger.warning(f"{layer_name} task not completed")
                except Exception as e:
                    logger.error(f"Error processing {layer_name}: {e}")
            
            # Process hazard results
            logger.info("Processing hazard results")
            for layer_name, task in hazard_tasks.items():
                try:
                    if task.done():
                        data = task.result()
                        if isinstance(data, Exception):
                            logger.error(f"Error in {layer_name} task: {data}")
                            continue
                            
                        if data and data.get('features'):
                            if layer_name == 'flood':
                                planning_info.flood_risk = True
                            elif layer_name == 'landslide':
                                planning_info.landslide_risk = True
                        else:
                            logger.debug(f"No {layer_name} hazard data found")
                    else:
                        logger.warning(f"{layer_name} task not completed")
                except Exception as e:
                    logger.error(f"Error processing {layer_name}: {e}")
    
    except Exception as e:
        logger.error(f"Error in get_all_planning_info: {e}")
        planning_info.error = str(e)
    
    end_time = time.time()
    logger.info(f"Total planning info retrieval took {end_time - start_time:.2f}s for {address}")
    return planning_info

async def get_planning_info(address: str, force_refresh: bool = False) -> PlanningInfo:
    """Get planning information for an address, with caching"""
    if not force_refresh:
        cached = await get_cached_result(address)
        if cached is not None:
            return PlanningInfo.model_validate({**cached, "source": "cache"})
    
    result = await get_all_planning_info(address)
    
    return result

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        test_address = "2 Martin Place, Sydney NSW 2000"
        result = await get_planning_info(test_address, force_refresh=True)
        
        if result.error:
            print(f"Error: {result.error}")
        else:
            print("\nPlanning Information Summary:")
            print(json.dumps(result.get_summary(), indent=2))
    
    asyncio.run(main())

