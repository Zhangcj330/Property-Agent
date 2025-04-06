from typing import List, Optional, Dict, Union
from google.cloud import firestore
from app.models import PropertySearchResponse, FirestoreProperty, InvestmentInfo, PlanningInfo
from app.services.image_processor import PropertyAnalysis
from app.config import settings
from datetime import datetime

class FirestoreService:
    def __init__(self):
        # Initialize with credentials dictionary
        self.db = firestore.Client.from_service_account_info(settings.FIREBASE_CONFIG)
        self.properties_collection = self.db.collection('properties')

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
            print(f"Error saving property: {str(e)}")
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
            print(f"Error updating analysis: {str(e)}")
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

    async def list_properties(self, 
        filters: Optional[Dict] = None, 
        limit: int = 10
    ) -> List[FirestoreProperty]:
        """List properties with optional filters"""
        pass

    async def delete_property(self, listing_id: str) -> bool:
        """Delete a property"""
        try:
            self.properties_collection.document(listing_id).delete()
            return True
        except Exception as e:
            print(f"Error deleting property: {str(e)}")
            raise 