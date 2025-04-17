import pandas as pd
import requests
import time
import asyncio
import aiohttp
from typing import Optional, Dict, Any
import backoff
from aiohttp import ClientSession
from asyncio import Semaphore
import json
from pathlib import Path
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

# Rate limiting settings
MAX_CONCURRENT_REQUESTS = 5
REQUEST_DELAY = 0.2  # 200ms between requests

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

class Location(BaseModel):
    address: str
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)

class PlanningInfo(BaseModel):
    location: Location
    # Planning data
    zone_name: Optional[str] = None
    height_limit: Optional[float] = None
    floor_space_ratio: Optional[float] = None
    min_lot_size: Optional[float] = None
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

def get_cached_result(address: str) -> Optional[Dict[str, Any]]:
    """Try to get cached result for an address"""
    try:
        cache_key = get_cache_key(address)
        cache_file = CACHE_DIR / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        with open(cache_file, 'r') as f:
            data = json.load(f)
            
        # Check if cache is still valid
        timestamp = data.get('timestamp', 0)
        if time.time() - timestamp > CACHE_DURATION.total_seconds():
            return None
            
        return data
        
    except Exception as e:
        print(f"Cache read error: {e}")
        return None

def clean_address(address: str) -> str:
    """Clean address by removing everything before and including the '/' character"""
    address = address.lower().strip()
    if '/' in address:
        address = address.split('/', 1)[1]
    return address.strip()

def get_coordinates(address: str) -> tuple[Optional[float], Optional[float]]:
    """Get coordinates for an address using OpenStreetMap"""
    try:
        address_cleaned = clean_address(address)
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={address_cleaned}"
        headers = {
            "User-Agent": 'MyApp/1.0 (zhangcj330@gmail.com)'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        results = response.json()
        if not results:
            return None, None
            
        return float(results[0]["lat"]), float(results[0]["lon"])
        
    except Exception as e:
        print(f"Error getting coordinates for {address}: {str(e)}")
        return None, None

@backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=3)
async def query_planning_layers(session: ClientSession, layer_index: int, lon: float, lat: float, semaphore: Semaphore) -> Optional[Dict[str, Any]]:
    """Query the EPI Primary Planning Layers MapServer"""
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
        
        async with session.get(query_url, params=params, timeout=30) as response:
            if response.status == 429:
                retry_after = int(response.headers.get('Retry-After', '60'))
                await asyncio.sleep(retry_after)
                return None
                
            data = await response.json()
            return data if 'features' in data else None

@backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=3)
async def query_hazard_layers(session: ClientSession, layer_index: int, lon: float, lat: float, semaphore: Semaphore) -> Optional[Dict[str, Any]]:
    """Query the Hazard MapServer"""
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
        
        async with session.get(query_url, params=params, timeout=30) as response:
            if response.status == 429:
                retry_after = int(response.headers.get('Retry-After', '60'))
                await asyncio.sleep(retry_after)
                return None
                
            data = await response.json()
            return data if 'features' in data else None

async def get_all_planning_info(address: str) -> PlanningInfo:
    """Get all planning information for a given address"""
    # Initialize with location
    location = Location(address=address)
    planning_info = PlanningInfo(location=location)
    
    try:
        # Get coordinates
        print(f"\nGetting coordinates for {address}...")
        lat, lon = get_coordinates(address)
        if lat is None or lon is None:
            planning_info.error = f"Could not get coordinates for address: {address}"
            return planning_info
            
        print(f"Found coordinates: {lat}, {lon}")
        location.latitude = lat
        location.longitude = lon
        
        # Setup API client
        semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
        timeout = aiohttp.ClientTimeout(total=120)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Get planning data
            for layer_name, layer_index in PLANNING_LAYERS.items():
                try:
                    print(f"\nQuerying {layer_name} layer...")
                    data = await query_planning_layers(session, layer_index, lon, lat, semaphore)
                    if data and data.get('features'):
                        print(f"Got {layer_name} data: {data['features'][0]['attributes']}")
                        attrs = data['features'][0]['attributes']
                        print(f"Processing {layer_name} attributes: {attrs}")
                        if layer_name == 'zoning':
                            planning_info.zone_name = attrs.get('LAY_CLASS')
                            print(f"Set zone_name to: {planning_info.zone_name}")
                        elif layer_name == 'height':
                            planning_info.height_limit = attrs.get('MAX_B_H')
                            print(f"Set height_limit to: {planning_info.height_limit}")
                        elif layer_name == 'fsr':
                            planning_info.floor_space_ratio = attrs.get('FSR')
                            print(f"Set floor_space_ratio to: {planning_info.floor_space_ratio}")
                        elif layer_name == 'lot_size':
                            planning_info.min_lot_size = attrs.get('MIN_LOT_SIZE')
                            print(f"Set min_lot_size to: {planning_info.min_lot_size}")
                        elif layer_name == 'heritage':
                            planning_info.is_heritage = True
                            print("Set is_heritage to: True")
                    else:
                        print(f"No {layer_name} data found")
                except Exception as e:
                    print(f"Error querying {layer_name}: {e}")
            
            # Get hazard data
            for layer_name, layer_index in HAZARD_LAYERS.items():
                try:
                    print(f"\nQuerying {layer_name} hazard layer...")
                    data = await query_hazard_layers(session, layer_index, lon, lat, semaphore)
                    if data and data.get('features'):
                        print(f"Got {layer_name} hazard data: {data['features'][0]['attributes']}")
                        if layer_name == 'flood':
                            planning_info.flood_risk = True
                        elif layer_name == 'landslide':
                            planning_info.landslide_risk = True
                    else:
                        print(f"No {layer_name} hazard data found")
                except Exception as e:
                    print(f"Error querying {layer_name}: {e}")
    
    except Exception as e:
        print(f"Error in get_all_planning_info: {e}")
        planning_info.error = str(e)
    
    return planning_info

async def get_planning_info(address: str, force_refresh: bool = False) -> PlanningInfo:
    """Get planning information for an address, with caching"""
    if not force_refresh:
        cached = get_cached_result(address)
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

