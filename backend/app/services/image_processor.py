import base64
import requests
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, HttpUrl
import json
from fastapi import HTTPException
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from app.config import settings

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

class QualityFactors(BaseModel):
    positive: List[str]
    negative: List[str]

class QualityAssessment(BaseModel):
    overall_score: float = Field(ge=0, le=10)
    condition: str
    maintenance_level: str
    build_quality: str
    environment: Environment
    factors: QualityFactors

class Style(BaseModel):
    architectural_style: str
    era: str
    design_elements: List[str]

class Features(BaseModel):
    interior: List[str]
    exterior: List[str]
    notable_amenities: List[str]

class RoomAnalysis(BaseModel):
    space_usage: str
    natural_light: str
    layout_quality: str

class PropertyAnalysis(BaseModel):
    style: Style
    features: Features
    quality_assessment: QualityAssessment
    room_analysis: RoomAnalysis

# Add API request/response models
class ImageAnalysisRequest(BaseModel):
    image_urls: List[HttpUrl] = Field(..., description="List of URLs of property images to analyze")

class ImageAnalysisResponse(BaseModel):
    analysis: Optional[PropertyAnalysis] = None
    error: Optional[str] = None

class ImageProcessor:
    def __init__(self):
        # Configure LLM client
        self.client = ChatOpenAI(
            api_key=settings.GEMINI_API_KEY,
            base_url=settings.BASE_URL,
            model="gemini-2.0-flash",
        )
        self.parser = JsonOutputParser(pydantic_object=PropertyAnalysis)
        
    def _encode_image_to_base64(self, image_url: str) -> str:
        """Convert image from URL to base64 string"""
        try:
            response = requests.get(str(image_url))
            response.raise_for_status()
            base64_encoded = base64.b64encode(response.content)
            return base64_encoded.decode('utf-8')
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to retrieve image from {image_url}. Error: {str(e)}"
            )

    async def analyze_property_image(self, request: ImageAnalysisRequest) -> ImageAnalysisResponse:
        """Analyze property images and return structured analysis"""
        try:
            # Convert images to base64
            base64_strings = [self._encode_image_to_base64(url) for url in request.image_urls]
            
            # Get the format instructions from the parser
            format_instructions = self.parser.get_format_instructions()
            
            # Prepare the prompt
            prompt = f"""Analyze this property image in detail. 
            Focus on real estate relevant details. Be specific about materials, finishes, and architectural elements.
            For quality score, consider factors like materials, maintenance, design coherence, and overall appeal.
            
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
            
            # Generate analysis using LLM
            response = self.client.invoke(messages)
            
            # Parse the response using the JsonOutputParser
            analysis = self.parser.parse(response.content)
            
            return ImageAnalysisResponse(analysis=analysis)
            
        except Exception as e:
            return ImageAnalysisResponse(error=str(e))
    