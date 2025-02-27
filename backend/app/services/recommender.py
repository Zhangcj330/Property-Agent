from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from app.models import UserPreferences, Property
from app.config import settings

class PropertyScore(BaseModel):
    score: float = Field(ge=0, le=1)
    reasoning: Dict[str, str]
    recommended: bool
    highlights: List[str]
    concerns: List[str]

class RecommendationAnalysis(BaseModel):
    property_id: str
    match_score: PropertyScore
    personalization_notes: str
    feature_alignment: Dict[str, bool]
    price_analysis: str

class PropertyRecommender:
    def __init__(self):
        self.client = ChatOpenAI(
            api_key=settings.GEMINI_API_KEY,
            base_url=settings.BASE_URL,
            model="gemini-2.0-flash",
        )
        self.parser = JsonOutputParser(pydantic_object=RecommendationAnalysis)

    async def get_recommendations(
        self,
        properties: List[Property],
        preferences: UserPreferences,
        limit: int = 10
    ) -> List[Property]:
        scored_properties = []
        
        for property in properties:
            analysis = await self._analyze_property_match(property, preferences)
            if analysis.match_score.recommended:
                scored_properties.append((property, analysis.match_score.score))
        
        # Sort by score and return top recommendations
        scored_properties.sort(key=lambda x: x[1], reverse=True)
        return [prop for prop, _ in scored_properties[:limit]]

    async def _analyze_property_match(
        self,
        property: Property,
        preferences: UserPreferences
    ) -> RecommendationAnalysis:
        try:
            # Prepare the prompt
            prompt = f"""Analyze how well this property matches the user's preferences.
            
            Property Details:
            - Price: ${property.price:,}
            - Location: {property.city}, {property.state}
            - Bedrooms: {property.bedrooms}
            - Bathrooms: {property.bathrooms}
            - Square Footage: {property.square_footage}
            - Type: {property.property_type}
            - Description: {property.description}
            
            User Preferences:
            - Max Price: ${preferences.max_price:,}
            - Min Bedrooms: {preferences.min_bedrooms}
            - Location: {preferences.location}
            - Property Type: {preferences.property_type or 'Any'}
            - Must-Have Features: {', '.join(preferences.must_have_features) if preferences.must_have_features else 'None specified'}
            
            Provide a detailed analysis of the match between the property and preferences.
            Consider price alignment, location desirability, feature match, and overall suitability.
            """

            # Get format instructions
            format_instructions = self.parser.get_format_instructions()
            
            messages = [
                HumanMessage(
                    content=f"{prompt}\n\n{format_instructions}"
                )
            ]
            
            # Generate analysis using LLM
            response = self.client.invoke(messages)
            
            # Parse the response
            analysis = self.parser.parse(response.content)
            return analysis
            
        except Exception as e:
            # Return a low-score analysis if something goes wrong
            return RecommendationAnalysis(
                property_id=property.id,
                match_score=PropertyScore(
                    score=0.0,
                    reasoning={"error": str(e)},
                    recommended=False,
                    highlights=[],
                    concerns=["Error in analysis"]
                ),
                personalization_notes="Analysis failed",
                feature_alignment={},
                price_analysis="Unable to analyze"
            ) 