from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
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
            model="gemini-2.0-flash",
        )
        self.parser = JsonOutputParser(pydantic_schema=List[PropertyRecommendation])

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
                    f"investment_info: {prop.investment_info}\n"
                    f"planning_info: {prop.planning_info}\n"
                )
                properties_summary.append(summary)

            prompt = (
                "Analyze these properties against the user's preferences and provide recommendations.\n\n"
                "Properties:\n"
                f"{chr(10).join(properties_summary)}\n\n"
                "User Preferences:\n"
                f"{preferences}\n\n"
                "For each property, provide the following structure:\n"
                "{\n"   
                "  property_id (string): The listing ID of the property.\n"
                "  score (number, between 0.0 and 1.0): The match quality score.\n"
                "  highlights (array of strings): List of positive features.\n"
                "  concerns (array of strings): List of negative features.\n"
                "  explanation (string): Brief explanation of the recommendation reasoning.\n"
                "}\n\n"
                "Please ensure you include all these fields for each property.\n"
                "Score each property from 0.00 (poor match) to 1.00 (perfect match) based on how well it meets the preferences.\n"
                "Include key highlights that match the user's preferences.\n"
                "List any concerns or mismatches with preferences.\n"
                "Provide a brief explanation about why each property is recommended or not.\n\n"
                f"Return analysis for all properties, sorted by match score (highest first)."
                "Only output the final response to the user, no other content."
                """Example:
[
  {
    "property_id": "listing-123456",
    "score": 0.92,
    "highlights": ["Great location", "Spacious backyard"],
    "concerns": ["Needs renovation"],
    "explanation": "This property matches most of the user's preferences, but requires some renovation."
  },
  ...
    ]
            """
            )

            messages = [
                HumanMessage(
                    content=f"{prompt}\n\n{self.parser.get_format_instructions()}"
                )
            ]
            
            # Get recommendations from LLM
            response = self.client.invoke(messages)
            
            # Handle both dictionary and parsed object scenarios
            try:
                # First try to parse using the parser
                recommendations = self.parser.parse(response.content)
                
                # Ensure we have a list of PropertyRecommendation objects
                if not isinstance(recommendations, list):
                    raise ValueError("Expected a list of recommendations")
                    
            except Exception as parsing_error:
                print(f"Parser error: {str(parsing_error)}")
                # If parsing fails, try to load as dictionary
                try:
                    import json
                    response_json = json.loads(response.content)
                    
                    # Handle different possible response structures
                    if isinstance(response_json, list):
                        # Direct list of recommendations
                        recommendations = []
                        for rec in response_json:
                            if isinstance(rec, dict) and "property_id" in rec:
                                recommendations.append(PropertyRecommendation(**rec))
                            else:
                                print(f"Skipping invalid recommendation: {rec}")
                    elif isinstance(response_json, dict) and "recommendations" in response_json:
                        # Nested recommendations in a dict
                        recommendations = []
                        for rec in response_json.get("recommendations", []):
                            if isinstance(rec, dict) and "property_id" in rec:
                                recommendations.append(PropertyRecommendation(**rec))
                            else:
                                print(f"Skipping invalid recommendation: {rec}")
                    else:
                        # Unknown format, return original properties
                        print(f"Unexpected response format: {response_json}")
                        return PropertyRecommendationResponse(properties=[])
                        
                except Exception as json_error:
                    print(f"JSON parsing error: {str(json_error)}")
                    # If all parsing fails, return original properties
                    return PropertyRecommendationResponse(properties=[])
            
            # Ensure recommendations is a valid list
            if not recommendations:
                print("No valid recommendations found")
                return PropertyRecommendationResponse(properties=[])
            
            # Create a map of property ids to their recommendation objects
            try:
                recommendation_map = {}
                for rec in recommendations:
                    if isinstance(rec, dict) and "property_id" in rec:
                        # If recommendation is a dict, use it directly
                        recommendation_map[rec["property_id"]] = PropertyRecommendation(**rec)
                    elif hasattr(rec, "property_id"):
                        # If recommendation is already a PropertyRecommendation object
                        recommendation_map[rec.property_id] = rec
                    else:
                        print(f"Skipping invalid recommendation format: {rec}")
                
                # Match original property objects with their recommendations
                recommended_properties = []
                for prop in properties:
                    if prop.listing_id in recommendation_map:
                        # Store the recommendation data with the property
                        setattr(prop, '_recommendation', recommendation_map[prop.listing_id])
                        recommended_properties.append(prop)
                
                # If no properties matched with recommendations, return original properties
                if not recommended_properties:
                    print("No properties matched with recommendations")
                    return PropertyRecommendationResponse(properties=[])
                
                # Sort by score (descending)
                recommended_properties.sort(key=lambda x: x._recommendation.score, reverse=True)
                
                for prop in recommended_properties:
                    print(f"Property {prop.listing_id}: score={prop._recommendation.score}")
                
                print("Sorted properties:", [f"{p.listing_id}:{p._recommendation.score}" for p in recommended_properties[:limit]])
                
                # Transform to the new response format
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
            except Exception as matching_error:
                print(f"Error matching properties with recommendations: {str(matching_error)}")
                # Return original properties in the new format without recommendations
                response_properties = []
                for prop in properties[:limit]:
                    response_properties.append(
                        PropertyWithRecommendation(
                            property=prop,
                            recommendation=PropertyRecommendationInfo(
                                score=0.0, 
                                highlights=["No recommendation available"],
                                concerns=["Unable to analyze property"],
                                explanation="An error occurred during recommendation processing"
                            )
                        )
                    )
                return PropertyRecommendationResponse(properties=response_properties)
            
        except Exception as e:
            print(f"Error in recommendation process: {str(e)}")
            # Return original properties in the new format without recommendations
            response_properties = []
            for prop in properties[:limit]:
                response_properties.append(
                    PropertyWithRecommendation(
                        property=prop,
                        recommendation=PropertyRecommendationInfo(
                            score=0.0, 
                            highlights=["No recommendation available"],
                            concerns=["Unable to analyze property"],
                            explanation="An error occurred during recommendation processing"
                        )
                    )
                )
            return PropertyRecommendationResponse(properties=response_properties) 
     