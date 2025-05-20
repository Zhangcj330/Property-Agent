import base64
import aiohttp
import asyncio
import time
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, HttpUrl
import json
from fastapi import HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class VisibleDefects(BaseModel):
    external_cracks: Literal["Absent", "Present"]
    structural_damage: Literal["Absent", "Present"]
    general_disrepair: Literal["Absent", "Present"]
    roof_gutter_damage: Literal["Absent", "Present"]

class ExteriorFeatures(BaseModel):
    building_materials: Literal["Brick", "Wood", "Concrete", "Other"]
    facade_condition: Literal["Well-maintained", "Moderate", "Poor"]
    external_fencing: Literal["None", "Partially Fenced", "Fully Fenced"]
    solar_panels: Literal["Yes", "No"]
    garden_condition: Literal["None", "Basic/Minimal", "Well-maintained"]
    parking_type: Literal["Garage", "Driveway", "Street", "None"]
    roof_material: Literal["Tile", "Metal", "Shingle", "Other"]

class InteriorFeatures(BaseModel):
    flooring_type: Literal["Mixed", "Tile", "Wood", "Carpet"]
    bathroom_condition: Literal["Modern/Updated", "Outdated"]
    kitchen_condition: Literal["Modern/Updated", "Outdated"]
    flooring_condition: Literal["Good", "Worn"]

class InteriorQualityStyle(BaseModel):
    design_style: Literal["Transitional", "Modern", "Traditional", "Other"]
    paint_decor: Literal["Neutral", "Bold", "Mixed"]
    lighting_natural_light: Literal["Bright", "Dim"]
    kitchen_style: Literal["Modern/Updated", "Outdated"]
    renovation_status: Literal["Partially Renovated", "Fully Renovated", "None"]

# Add Pydantic models
class Environment(BaseModel):
    privacy: str
    noise_exposure: str
    lighting_conditions: str
    sustainability_features: List[str]
    road_proximity: str
    pole_or_line_of_sight: str
    land_flatness: str
    greenery: str


class PropertyAnalysis(BaseModel):
    visible_defects: VisibleDefects
    exterior_features: ExteriorFeatures
    interior_features: InteriorFeatures
    interior_quality_style: InteriorQualityStyle
    environment: Environment
    property_description_on_user_preference: Optional[str] = Field(None, description="property description based on the user's personal preferences")

# Add API request/response models
class ImageAnalysisRequest(BaseModel):
    image_urls: List[HttpUrl] = Field(..., description="List of URLs of property images to analyze")
    preferences: Optional[str] = Field(None, description="User personal preferences for the property")

class ImageProcessor:
    def __init__(self):
        # Configure LLM client
        self.client = ChatGoogleGenerativeAI(
            api_key=settings.GEMINI_API_KEY,
            base_url=settings.BASE_URL,
            model="gemini-2.0-flash",
        )
        self.parser = JsonOutputParser(pydantic_object=PropertyAnalysis)
        
    async def _encode_image_to_base64(self, image_url: str) -> str:
        """Convert image from URL to base64 string using aiohttp (async)"""
        try:
            download_start = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(str(image_url), timeout=10) as response:
                    if response.status != 200:
                        logger.error(f"Image download error: HTTP {response.status} for {image_url}")
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to retrieve image from {image_url}. Status: {response.status}"
                        )
                    content = await response.read()
            download_end = time.time()
            logger.debug(f"Image download took {download_end - download_start:.2f}s for {image_url}")
            
            encode_start = time.time()
            base64_encoded = base64.b64encode(content)
            result = base64_encoded.decode('utf-8')
            encode_end = time.time()
            logger.debug(f"Base64 encoding took {encode_end - encode_start:.2f}s for image size {len(content)/1024:.1f}KB")
            
            return result
        except aiohttp.ClientError as e:
            logger.error(f"Image download failed: {str(e)} for {image_url}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to retrieve image from {image_url}. Error: {str(e)}"
            )

    async def analyze_property_image(self, request: ImageAnalysisRequest, preferences: Optional[str] = None) -> PropertyAnalysis:
        """Analyze property images and return structured analysis"""
        overall_start = time.time()
        try:
            
            # Convert images to base64 concurrently
            base64_tasks = [self._encode_image_to_base64(url) for url in request.image_urls]
            base64_strings = await asyncio.gather(*base64_tasks)
            
            # Get the format instructions from the parser
            format_instructions = self.parser.get_format_instructions()
            
            # Prepare the prompt
            prompt = f"""Analyze property images in detail. 
            Focus on real estate relevant details. Be specific about materials, finishes, and architectural elements.
            For quality score, consider factors like materials, maintenance, design coherence, and overall appeal.
            if user has provided thier preference: {request.preferences}
            generate the property_description_on_user_preference. 
            
            {format_instructions}"""

            # Prepare messages for the LLM with multiple images
            message_content = [{"type": "text", "text": prompt}]
            
            # Add all images to the content
            for base64_string in base64_strings:
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_string}"
                    }
                })

            messages = [HumanMessage(content=message_content)]
            
            # Use ainvoke for asynchronous invocation
            response = await self.client.ainvoke(messages)
            # Parse the response using the JsonOutputParser
            analysis = self.parser.parse(response.content)

            return analysis
            
        except Exception as e:
            overall_end = time.time()
            logger.error(f"Image analysis failed after {overall_end - overall_start:.2f}s: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
