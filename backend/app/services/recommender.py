from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from app.models import UserPreferences, FirestoreProperty
from app.config import settings

class PropertyRecommendation(BaseModel):
    property_id: str
    score: float = Field(ge=0, le=1)
    highlights: List[str]
    concerns: List[str]
    explanation: str


class PropertyRecommender:
    def __init__(self):
        self.client = ChatOpenAI(
            api_key=settings.GEMINI_API_KEY,
            base_url=settings.BASE_URL,
            model="gemini-2.0-flash",
        )
        self.parser = JsonOutputParser(pydantic_schema=List[PropertyRecommendation])

    async def get_recommendations(
        self,
        properties: List[FirestoreProperty],
        preferences: UserPreferences,
        limit: int = 1
    ) -> List[FirestoreProperty]:
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
                '  "property_id": "listing ID of the property",\n'
                '  "score": float between 0.0-1.0 representing match quality,\n'
                '  "highlights": ["list", "of", "positive features"],\n'
                '  "concerns": ["list", "of", "negative features"],\n'
                '  "explanation": "Brief explanation of the recommendation reasoning"\n'
                "}\n\n"
                "Please ensure you include all these fields for each property.\n"
                "Score each property from 0.0 (poor match) to 1.0 (perfect match) based on how well it meets the preferences.\n"
                "Include key highlights that match the user's preferences.\n"
                "List any concerns or mismatches with preferences.\n"
                "Provide a brief explanation about why each property is recommended or not.\n\n"
                f"Return analysis for all properties, sorted by match score (highest first)."
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
                        return properties[:limit]
                        
                except Exception as json_error:
                    print(f"JSON parsing error: {str(json_error)}")
                    # If all parsing fails, return original properties
                    return properties[:limit]
            
            # Ensure recommendations is a valid list
            if not recommendations:
                print("No valid recommendations found")
                return properties[:limit]
            
            # Create a map of property ids to their recommendation objects
            try:
                recommendation_map = {rec.property_id: rec for rec in recommendations}
                
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
                    return properties[:limit]
                
                # Sort by score (descending)
                recommended_properties.sort(
                    key=lambda x: getattr(getattr(x, '_recommendation', None), 'score', 0) if hasattr(x, '_recommendation') else 0,
                    reverse=True
                )
                
                return recommended_properties[:limit]
            except Exception as matching_error:
                print(f"Error matching properties with recommendations: {str(matching_error)}")
                return properties[:limit]
            
        except Exception as e:
            print(f"Error in recommendation process: {str(e)}")
            # Return original properties if analysis fails
            return properties[:limit] 