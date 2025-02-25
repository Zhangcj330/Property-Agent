from typing import List, Dict
from sqlalchemy.orm import Session
from ..models import UserPreferences, Property

class PropertyRecommender:
    def __init__(self):
        self.feature_weights = {
            "price_match": 0.3,
            "location_match": 0.3,
            "features_match": 0.2,
            "property_type_match": 0.2
        }

    async def get_recommendations(
        self,
        db: Session,
        preferences: UserPreferences,
        limit: int = 10
    ) -> List[Property]:
        # Get initial candidates
        candidates = self._get_candidates(db, preferences)
        
        # Score and rank properties
        scored_properties = []
        for property in candidates:
            score = self._calculate_score(property, preferences)
            scored_properties.append((property, score))
        
        # Sort by score and return top recommendations
        scored_properties.sort(key=lambda x: x[1], reverse=True)
        return [prop for prop, score in scored_properties[:limit]]

    def _calculate_score(self, property: Property, preferences: UserPreferences) -> float:
        score = 0.0
        
        # Price match
        if preferences.max_price >= property.price >= (preferences.min_price or 0):
            score += self.feature_weights["price_match"]
        
        # Location match
        if preferences.location.lower() in property.city.lower():
            score += self.feature_weights["location_match"]
        
        # Property type match
        if preferences.property_type and preferences.property_type.lower() == property.property_type.lower():
            score += self.feature_weights["property_type_match"]
        
        # Features match
        if preferences.must_have_features:
            feature_match_ratio = len(
                set(preferences.must_have_features) & set(f.name for f in property.features)
            ) / len(preferences.must_have_features)
            score += self.feature_weights["features_match"] * feature_match_ratio
        
        return score

    def _get_candidates(self, db: Session, preferences: UserPreferences) -> List[Property]:
        query = db.query(Property)
        
        if preferences.min_price:
            query = query.filter(Property.price >= preferences.min_price)
        if preferences.max_price:
            query = query.filter(Property.price <= preferences.max_price)
        if preferences.min_bedrooms:
            query = query.filter(Property.bedrooms >= preferences.min_bedrooms)
        
        return query.all() 