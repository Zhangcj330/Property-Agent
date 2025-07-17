from typing import List, Dict, Optional
import re
import json
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from app.models import UserPreferences, FirestoreProperty, PropertyRecommendationResponse, PropertyWithRecommendation, PropertyRecommendationInfo
from app.config import settings

class PropertyRecommendation(BaseModel):
    property_id: str
    score: float = Field(ge=0, le=1)
    highlights: List[str]
    concerns: List[str]
    explanation: str


class PropertyRecommender:
    def __init__(self):
        self.client = ChatGoogleGenerativeAI(
            api_key=settings.GEMINI_API_KEY,
            base_url=settings.BASE_URL,
            model="gemini-2.5-flash",
        )
        self.parser = JsonOutputParser(pydantic_schema=List[PropertyRecommendation])

    def _extract_json_from_markdown(self, content: str) -> str:
        """Extract JSON content from markdown code blocks or return original content."""
        # Try to extract from ```json ... ``` blocks
        json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Try to extract from ``` ... ``` blocks (any code block)
        code_match = re.search(r'```\s*\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Try to find JSON array pattern in text
        array_match = re.search(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)
        if array_match:
            return array_match.group(0).strip()
        
        # Return original content if no patterns found
        return content.strip()

    def _create_fallback_recommendations(self, properties: List[FirestoreProperty]) -> List[PropertyRecommendation]:
        """Create fallback recommendations when LLM parsing fails"""
        recommendations = []
        for i, prop in enumerate(properties):
            # Simple scoring based on available data
            score = 0.5  # Default neutral score
            highlights = []
            concerns = []
            
            # Basic analysis based on available data
            if prop.investment_info:
                if prop.investment_info.rental_yield and prop.investment_info.rental_yield > 5:
                    highlights.append(f"Good rental yield: {prop.investment_info.rental_yield}%")
                    score += 0.1
                if prop.investment_info.capital_gain and prop.investment_info.capital_gain > 5:
                    highlights.append(f"Positive growth forecast: {prop.investment_info.capital_gain}%")
                    score += 0.1
            
            if prop.analysis:
                if hasattr(prop.analysis, 'overall_condition') and prop.analysis.overall_condition:
                    if 'good' in prop.analysis.overall_condition.lower():
                        highlights.append("Property appears to be in good condition")
                        score += 0.1
                    elif 'poor' in prop.analysis.overall_condition.lower():
                        concerns.append("Property may need maintenance")
                        score -= 0.1
            
            if not highlights:
                highlights = ["Standard property features"]
            if not concerns:
                concerns = ["Standard considerations apply"]
            
            # Ensure score is within bounds
            score = max(0.0, min(1.0, score))
            
            recommendations.append(PropertyRecommendation(
                property_id=prop.listing_id,
                score=score,
                highlights=highlights,
                concerns=concerns,
                explanation=f"Property {i+1} with basic analysis. Score based on available data."
            ))
        
        return recommendations

    async def get_recommendations(
        self,
        properties: List[FirestoreProperty],
        preferences: UserPreferences,
        limit: int = 2
    ) -> PropertyRecommendationResponse:
        try:
            # Prepare the batch analysis prompt
            properties_summary = []
            for idx, prop in enumerate(properties):
                summary = (
                    f"Property {idx + 1}:\n"
                    f"ID: {prop.listing_id}\n"
                    f"Price: ${prop.basic_info.price_value if prop.basic_info.price_is_numeric else 'Contact agent'}\n"
                    f"Address: {prop.basic_info.full_address}\n"
                    f"Bedrooms: {prop.basic_info.bedrooms_count}\n"
                    f"Bathrooms: {prop.basic_info.bathrooms_count}\n"
                    f"Type: {prop.basic_info.property_type}\n"
                    f"Image Analysis: {str(prop.analysis) if prop.analysis else 'No analysis'}\n"
                    f"Investment Info: {prop.investment_info}\n"
                    f"Planning Info: {prop.planning_info}\n"
                )
                properties_summary.append(summary)

            # Enhanced system message for better output consistency
            system_prompt = """You are a property recommendation expert. You must analyze properties and return ONLY a valid JSON array.

CRITICAL: Your response must be a valid JSON array starting with [ and ending with ]. No other text, explanations, or formatting.

Each property analysis must follow this EXACT structure:
{
  "property_id": "exact_listing_id_from_input",
  "score": 0.75,
  "highlights": ["specific positive feature 1", "specific positive feature 2"],
  "concerns": ["specific concern 1", "specific concern 2"],
  "explanation": "Brief reason for recommendation"
}

Rules:
- Score: 0.0 to 1.0 (use decimals like 0.75, 0.82)
- Always include at least 1 highlight and 1 concern
- Keep explanations under 100 characters
- Return all properties analyzed
- Sort by score (highest first)"""

            user_prompt = f"""Analyze these properties against user preferences and return JSON array only.

Properties:
{chr(10).join(properties_summary)}

User Preferences: {preferences}

Return ONLY the JSON array with no additional text:"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Get recommendations from LLM with retry logic
            max_retries = 2
            recommendations = None
            
            for attempt in range(max_retries + 1):
                try:
                    response = self.client.invoke(messages)
                    content = response.content.strip()
                    
                    # Log the raw response for debugging
                    print(f"LLM Response (attempt {attempt + 1}): {content[:200]}...")
                    
                    # Try multiple parsing strategies
                    parsing_strategies = [
                        lambda: self.parser.parse(content),
                        lambda: self.parser.parse(self._extract_json_from_markdown(content)),
                        lambda: json.loads(content),
                        lambda: json.loads(self._extract_json_from_markdown(content))
                    ]
                    
                    for strategy in parsing_strategies:
                        try:
                            result = strategy()
                            if isinstance(result, list) and len(result) > 0:
                                # Validate each recommendation has required fields
                                valid_recommendations = []
                                for rec in result:
                                    if isinstance(rec, dict):
                                        try:
                                            valid_rec = PropertyRecommendation(**rec)
                                            valid_recommendations.append(valid_rec)
                                        except Exception as e:
                                            print(f"Invalid recommendation structure: {e}")
                                            continue
                                    elif hasattr(rec, 'property_id'):
                                        valid_recommendations.append(rec)
                                
                                if valid_recommendations:
                                    recommendations = valid_recommendations
                                    break
                        except Exception as e:
                            continue
                    
                    if recommendations:
                        break
                        
                except Exception as e:
                    print(f"LLM call attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries:
                        break
            
            # If all parsing attempts failed, use fallback
            if not recommendations:
                print("All LLM parsing attempts failed, using fallback recommendations")
                recommendations = self._create_fallback_recommendations(properties)
            
            # Create a map of property ids to their recommendation objects
            recommendation_map = {}
            for rec in recommendations:
                if hasattr(rec, "property_id"):
                    recommendation_map[rec.property_id] = rec
                elif isinstance(rec, dict) and "property_id" in rec:
                    recommendation_map[rec["property_id"]] = PropertyRecommendation(**rec)
            
            # Match original property objects with their recommendations
            recommended_properties = []
            for prop in properties:
                if prop.listing_id in recommendation_map:
                    setattr(prop, '_recommendation', recommendation_map[prop.listing_id])
                    recommended_properties.append(prop)
            
            # If no properties matched, use fallback approach
            if not recommended_properties:
                print("No properties matched with recommendations, using fallback")
                fallback_recs = self._create_fallback_recommendations(properties)
                for i, prop in enumerate(properties):
                    if i < len(fallback_recs):
                        setattr(prop, '_recommendation', fallback_recs[i])
                        recommended_properties.append(prop)
            
            # Sort by score (descending)
            recommended_properties.sort(key=lambda x: x._recommendation.score, reverse=True)
            
            # Transform to the response format
            response_properties = []
            for prop in recommended_properties[:limit]:
                rec = prop._recommendation
                response_properties.append(
                    PropertyWithRecommendation(
                        property=prop,
                        recommendation=PropertyRecommendationInfo(
                            score=rec.score,
                            highlights=rec.highlights,
                            concerns=rec.concerns,
                            explanation=rec.explanation
                        )
                    )
                )
            
            return PropertyRecommendationResponse(properties=response_properties)
            
        except Exception as e:
            print(f"Error in recommendation process: {str(e)}")
            # Return fallback recommendations
            fallback_recs = self._create_fallback_recommendations(properties)
            response_properties = []
            for i, prop in enumerate(properties[:limit]):
                if i < len(fallback_recs):
                    rec = fallback_recs[i]
                else:
                    rec = PropertyRecommendation(
                        property_id=prop.listing_id,
                        score=0.5,
                        highlights=["Standard property"],
                        concerns=["Analysis unavailable"],
                        explanation="Fallback analysis"
                    )
                
                response_properties.append(
                    PropertyWithRecommendation(
                        property=prop,
                        recommendation=PropertyRecommendationInfo(
                            score=rec.score,
                            highlights=rec.highlights,
                            concerns=rec.concerns,
                            explanation=rec.explanation
                        )
                    )
                )
            return PropertyRecommendationResponse(properties=response_properties) 
     