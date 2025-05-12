from typing import List, Optional, Dict, Union, Tuple
from google.cloud import firestore
from app.models import PropertySearchResponse, FirestoreProperty, InvestmentInfo, PlanningInfo, PropertyWithRecommendation
from app.services.image_processor import PropertyAnalysis
from app.config import settings
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

class FirestoreService:
    def __init__(self):
        # Don't initialize Firebase connections immediately
        self._db = None
        self._properties_collection = None
        self._saved_properties_collection = None
        self._feedback_collection = None
    
    @property
    def db(self):
        if self._db is None:
            # Initialize with credentials dictionary only when needed
            logger.info("Initializing Firestore connection")
            self._db = firestore.Client.from_service_account_info(settings.FIREBASE_CONFIG)
        return self._db
    
    @property
    def properties_collection(self):
        if self._properties_collection is None:
            self._properties_collection = self.db.collection('properties')
        return self._properties_collection
    
    @property
    def saved_properties_collection(self):
        if self._saved_properties_collection is None:
            self._saved_properties_collection = self.db.collection('saved_properties')
        return self._saved_properties_collection
    
    @property
    def feedback_collection(self):
        if self._feedback_collection is None:
            self._feedback_collection = self.db.collection('feedback')
        return self._feedback_collection

    async def save_property(self, property_data: Union[PropertySearchResponse, FirestoreProperty]) -> str:
        """Save or update a property listing
        
        Args:
            property_data: Either a PropertySearchResponse or FirestoreProperty object
        
        Returns:
            str: The listing ID of the saved property
        """
        try:
            # Convert to FirestoreProperty if needed
            firestore_property = (
                property_data if isinstance(property_data, FirestoreProperty)
                else FirestoreProperty.from_search_response(property_data)
            )
            
            # Use listing_id as document ID
            doc_ref = self.properties_collection.document(firestore_property.listing_id)
            
            # Check if document exists to handle updates
            doc = doc_ref.get()
            if doc.exists:
                # Get existing data
                existing_data = doc.to_dict()
                
                # Update timestamp
                firestore_property.metadata.updated_at = datetime.now()
                
                # Preserve existing data if not in new data
                if existing_data.get('analysis') and not firestore_property.analysis:
                    firestore_property.analysis = existing_data['analysis']
                if existing_data.get('investment_info') and not any(vars(firestore_property.investment_info).values()):
                    firestore_property.investment_info = InvestmentInfo(**existing_data['investment_info'])
                if existing_data.get('planning_info') and not any(vars(firestore_property.planning_info).values()):
                    firestore_property.planning_info = PlanningInfo(**existing_data['planning_info'])
            
            # Save to Firestore
            doc_ref.set(firestore_property.model_dump())
            return firestore_property.listing_id
            
        except Exception as e:
            logger.error(f"Error saving property: {str(e)}")
            raise

    async def update_property_analysis(self, listing_id: str, analysis: PropertyAnalysis) -> bool:
        """Update property analysis"""
        try:
            doc_ref = self.properties_collection.document(listing_id)
            # Update only the analysis field and updated_at timestamp
            analysis_dict = analysis.model_dump() if hasattr(analysis, 'model_dump') else analysis
            doc_ref.update({
                "analysis": analysis_dict,
                "updated_at": datetime.now()
            })
            return True
        except Exception as e:
            logger.error(f"Error updating analysis: {str(e)}")
            raise

    async def get_property(self, listing_id: str) -> Optional[FirestoreProperty]:
        """Retrieve a property with its analysis by listing ID"""
        try:
            doc_ref = self.properties_collection.document(listing_id)
            doc = doc_ref.get()
            if doc.exists:
                # Convert Firestore data to FirestoreProperty
                firestore_property = FirestoreProperty(**doc.to_dict())
                # Convert to analysis response
                return firestore_property
            return None
        except Exception as e:
            print(f"Error retrieving property: {str(e)}")
            raise

    async def delete_property(self, listing_id: str) -> bool:
        """Delete a property"""
        try:
            self.properties_collection.document(listing_id).delete()
            return True
        except Exception as e:
            print(f"Error deleting property: {str(e)}")
            raise

    async def save_property_to_session(
        self, 
        session_id: str, 
        property_with_recommendation: PropertyWithRecommendation
    ) -> bool:
        """Save a property to a session's saved properties
        
        Args:
            session_id: The session ID
            property_with_recommendation: The property with its recommendation info
            
        Returns:
            bool: True if successful
        """
        try:
            # Get or create saved properties document
            doc_ref = self.saved_properties_collection.document(session_id)
            doc = doc_ref.get()
            
            if doc.exists:
                # Update existing saved properties
                saved_data = doc.to_dict()
                saved_properties = [
                    PropertyWithRecommendation(**p) 
                    for p in saved_data.get('properties', [])
                ]
                
                # Check if property already exists
                for i, prop in enumerate(saved_properties):
                    if prop.property.listing_id == property_with_recommendation.property.listing_id:
                        # Update existing property
                        saved_properties[i] = property_with_recommendation
                        break
                else:
                    # Add new property
                    saved_properties.append(property_with_recommendation)
                
                # Update document
                doc_ref.update({
                    'properties': [p.model_dump() for p in saved_properties],
                    'updated_at': datetime.now()
                })
            else:
                # Create new saved properties document
                doc_ref.set({
                    'session_id': session_id,
                    'properties': [property_with_recommendation.model_dump()],
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                })
            
            return True
            
        except Exception as e:
            print(f"Error saving property to session: {str(e)}")
            raise

    async def get_saved_properties(self, session_id: str) -> List[PropertyWithRecommendation]:
        """Get saved properties for a session
        
        Args:
            session_id: The session ID
            
        Returns:
            List[PropertyWithRecommendation]: List of saved properties with recommendations
        """
        try:
            # Get saved properties document
            doc_ref = self.saved_properties_collection.document(session_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                return []
                
            saved_data = doc.to_dict()
            return [
                PropertyWithRecommendation(**p) 
                for p in saved_data.get('properties', [])
            ]
            
        except Exception as e:
            print(f"Error getting saved properties: {str(e)}")
            raise

    async def remove_saved_property(self, session_id: str, property_id: str) -> bool:
        """Remove a property from a session's saved properties
        
        Args:
            session_id: The session ID
            property_id: The property listing ID to remove
            
        Returns:
            bool: True if successful
        """
        try:
            doc_ref = self.saved_properties_collection.document(session_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                return False
                
            saved_data = doc.to_dict()
            saved_properties = [
                PropertyWithRecommendation(**p) 
                for p in saved_data.get('properties', [])
            ]
            
            # Filter out the property to remove
            updated_properties = [
                p for p in saved_properties 
                if p.property.listing_id != property_id
            ]
            
            if len(updated_properties) != len(saved_properties):
                # Property was found and removed
                doc_ref.update({
                    'properties': [p.model_dump() for p in updated_properties],
                    'updated_at': datetime.now()
                })
                return True
            
            return False
            
        except Exception as e:
            print(f"Error removing saved property: {str(e)}")
            raise

    async def save_feedback(self, 
                           feedback_text: str,
                           feedback_type: str = 'general',
                           session_id: Optional[str] = None,
                           screenshot_url: Optional[str] = None) -> str:
        """Save user feedback to Firestore
        
        Args:
            feedback_text: The feedback text content
            feedback_type: The type of feedback (default: 'general')
            session_id: Optional session ID the feedback is associated with
            screenshot_url: Optional URL to a screenshot
            
        Returns:
            str: The created feedback ID
        """
        try:
            feedback_id = str(uuid.uuid4())
            doc_ref = self.feedback_collection.document(feedback_id)
            
            feedback_data = {
                'id': feedback_id,
                'text': feedback_text,
                'type': feedback_type,
                'session_id': session_id,
                'has_screenshot': bool(screenshot_url),
                'screenshot_url': screenshot_url,
                'created_at': datetime.now(),
            }
            
            doc_ref.set(feedback_data)
            return feedback_id
            
        except Exception as e:
            print(f"Error saving feedback: {str(e)}")
            raise 