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

class BatchRecommendationResult(BaseModel):
    recommendations: List[PropertyRecommendation]
    explanation: str
    preference_analysis: Dict[str, str]

class PropertyRecommender:
    def __init__(self):
        self.client = ChatOpenAI(
            api_key=settings.GEMINI_API_KEY,
            base_url=settings.BASE_URL,
            model="gemini-2.0-flash",
        )
        self.parser = JsonOutputParser(pydantic_object=BatchRecommendationResult)

    async def get_recommendations(
        self,
        properties: List[FirestoreProperty],
        preferences: UserPreferences,
        limit: int = 10
    ) -> List[FirestoreProperty]:
        try:
            # Prepare the batch analysis prompt
            properties_summary = []
            for idx, prop in enumerate(properties):
                p = prop.properties_search
                summary = (
                    f"Property {idx + 1}:\n"
                    f"ID: {p.listing_id}\n"
                    f"Price: {p.price}\n"
                    f"Address: {p.address}\n"
                    f"Bedrooms: {p.bedrooms}\n"
                    f"Bathrooms: {p.bathrooms}\n"
                    f"Type: {p.property_type}\n"
                    f"Image Analysis: {prop.image_analysis.summary if prop.image_analysis else 'No analysis'}\n"
                )
                properties_summary.append(summary)

            prompt = (
                "Analyze these properties against the user's preferences and provide recommendations.\n\n"
                "Properties:\n"
                f"{chr(10).join(properties_summary)}\n\n"
                "User Preferences:\n"
                f"{preferences}\n\n"
                "Provide:\n"
                "1. A score (0-1) for each property based on preference match\n"
                "2. Key highlights for each recommended property\n"
                "3. Any concerns or mismatches\n"
                "4. Overall explanation of recommendations\n"
                "5. Analysis of how preferences were applied\n\n"
                f"Return the top {limit} most suitable properties."
            )

            messages = [
                HumanMessage(
                    content=f"{prompt}\n\n{self.parser.get_format_instructions()}"
                )
            ]
            
            # Get batch analysis from LLM
            response = self.client.invoke(messages)
            analysis = self.parser.parse(response.content)
            
            # Sort properties by score and get top recommendations
            property_scores = {
                rec.property_id: rec.score 
                for rec in analysis.recommendations
            }
            
            # Filter and sort properties based on scores
            recommended_properties = [
                prop for prop in properties 
                if prop.properties_search.listing_id in property_scores
            ]
            recommended_properties.sort(
                key=lambda x: property_scores[x.properties_search.listing_id],
                reverse=True
            )
            
            return recommended_properties[:limit]
            
        except Exception as e:
            print(f"Error in batch recommendation: {str(e)}")
            # Return original properties if analysis fails
            return properties[:limit] 