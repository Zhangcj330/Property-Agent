from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple, Any, Union
from typing_extensions import TypedDict, Literal
from datetime import datetime
import re

class UserPreference(TypedDict):
    """User preference for property search"""
    preference: Optional[str]
    importance: float 

class UserPreferences(TypedDict):
    """User preferences for property search
    The first element is the preference value, the second is the importance
    """
    Features: UserPreference = UserPreference(preference=None, importance=0.5)
    Style: UserPreference = UserPreference(preference=None, importance=0.5)
    Condition: UserPreference = UserPreference(preference=None, importance=0.5)
    Environment: UserPreference = UserPreference(preference=None, importance=0.5)
    SchoolDistrict: UserPreference = UserPreference(preference=None, importance=0.5)
    Community: UserPreference = UserPreference(preference=None, importance=0.5)
    Transport: UserPreference = UserPreference(preference=None, importance=0.5)        
    Location: UserPreference = UserPreference(preference=None, importance=0.5)
    Investment: UserPreference = UserPreference(preference=None, importance=0.5)

# Request model for property search
class PropertySearchRequest(TypedDict):
    location: Optional[List[str]] = Field(None, description="List of locations to search, format: STATE-SUBURB-POSTCODE")
    min_price: Optional[float] = Field(None, description="Minimum price")
    max_price: Optional[float] = Field(None, description="Maximum price")
    min_bedrooms: Optional[int] = Field(None, description="Minimum number of bedrooms")
    min_bathrooms: Optional[int] = Field(None, description="Minimum number of bathrooms")
    property_type: Optional[List[str]] = Field(None, description="List of property types (house, apartment, unit, townhouse, villa, rural)")
    car_parks: Optional[int] = Field(None, description="Number of car parks")
    land_size_from: Optional[float] = Field(None, description="Minimum land size in sqm")
    land_size_to: Optional[float] = Field(None, description="Maximum land size in sqm")
    # geo_location: Optional[Tuple[float, float]] = Field(None, description="Geographical location as [latitude, longitude]")

# Response model for property search
class PropertySearchResponse(BaseModel):
    listing_id: Optional[str]
    price: str
    address: str
    bedrooms: str
    bathrooms: str
    car_parks: Optional[str]
    land_size: Optional[str]
    property_type: Optional[str]
    inspection_date: Optional[str]
    image_urls: Optional[List[str]]
    agent_name: Optional[str]

# Nested models for FirestoreProperty
class PropertyBasicInfo(BaseModel):
    """Basic property information"""
    price_value: Optional[float] = None
    price_is_numeric: bool = False
    
    # Address components
    full_address: str
    street_address: Optional[str] = None
    suburb: Optional[str] = None
    state: Optional[str] = None
    postcode: Optional[str] = None
    
    bedrooms_count: Optional[int] = None
    bathrooms_count: Optional[float] = None
    car_parks: Optional[str] = None
    land_size: Optional[str] = None
    property_type: Optional[str] = None

class PropertyMedia(BaseModel):
    """Property media information"""
    image_urls: Optional[List[str]] = None

class AgentInfo(BaseModel):
    """Agent information"""
    agent_name: Optional[str] = None

class PropertyMetadata(BaseModel):
    """Property metadata"""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    source: Optional[str] = "scraper"
    status: Optional[str] = "active"  # active, sold, withdrawn, etc.
    session_id: Optional[str] = None  # Added for session tracking

class InvestmentInfo(BaseModel):
    """Investment metrics for a property"""
    rental_yield: Optional[float] = None  # Annual rental yield as percentage
    capital_gain: Optional[float] = None  # 1-year capital gain as percentage
    current_price: Optional[float] = None  # Current median price
    weekly_rent: Optional[float] = None   # Current weekly rent

class PlanningInfo(BaseModel):
    """Planning information for a property"""
    zone_name: Optional[str] = None
    height_limit: Optional[float] = None
    floor_space_ratio: Optional[float] = None
    min_lot_size: Optional[float] = None
    is_heritage: bool = False
    flood_risk: bool = False
    landslide_risk: bool = False

class PropertyEvents(BaseModel):
    """Property events and important dates"""
    inspection_date: Optional[str] = None
    auction_date: Optional[str] = None

# Firestore optimized property model with modular structure
class FirestoreProperty(BaseModel):
    """Model optimized for Firestore storage with modular structure"""
    # Core identification
    listing_id: str
    
    # Structured property information
    basic_info: PropertyBasicInfo = Field(default_factory=PropertyBasicInfo)
    media: PropertyMedia = Field(default_factory=PropertyMedia)
    agent: AgentInfo = Field(default_factory=AgentInfo)
    events: PropertyEvents = Field(default_factory=PropertyEvents)
    metadata: PropertyMetadata = Field(default_factory=PropertyMetadata)
    
    # Analysis data (can be null)
    analysis: Optional[Dict[str, Any]] = None
    
    # Investment and planning information
    investment_info: InvestmentInfo = Field(default_factory=InvestmentInfo)
    planning_info: PlanningInfo = Field(default_factory=PlanningInfo)
    
    @classmethod
    def from_search_response(cls, response: PropertySearchResponse):
        """Convert PropertySearchResponse to FirestoreProperty"""
        # 提取价格数值
        price_value = None
        price_is_numeric = False
        
        if response.price:
            # 处理价格范围 "$X to $Y"
            price_range_match = re.search(r'\$(\d+(?:,\d+)*)\s+to\s+\$(\d+(?:,\d+)*)', response.price)
            if price_range_match:
                try:
                    min_price = float(price_range_match.group(1).replace(',', ''))
                    max_price = float(price_range_match.group(2).replace(',', ''))
                    price_value = (min_price + max_price) / 2  # 取中间值
                    price_is_numeric = True
                except (ValueError, AttributeError):
                    price_value = None
            else:
                # 提取$后面的数字
                price_match = re.search(r'\$(\d+(?:,\d+)*)', response.price)
                if price_match:
                    try:
                        price_value = float(price_match.group(1).replace(',', ''))
                        price_is_numeric = True
                    except (ValueError, AttributeError):
                        price_value = None
        
        # 提取卧室数量
        try:
            bedrooms_count = int(response.bedrooms)
        except (ValueError, AttributeError):
            bedrooms_count = None
            
        # 提取浴室数量
        try:
            bathrooms_count = float(response.bathrooms)
        except (ValueError, AttributeError):
            bathrooms_count = None
        
        # 解析地址组件
        full_address = response.address
        street_address = None
        suburb = None
        state = None
        postcode = None
        
        # 尝试解析澳大利亚地址格式
        if full_address:
            # 按逗号分割
            address_parts = [part.strip() for part in full_address.split(',')]
            
            # 最后一部分通常包含州和邮编
            if len(address_parts) > 1:
                last_part = address_parts[-1]
                # 尝试提取州和邮编
                state_postcode = last_part.split()
                if len(state_postcode) >= 2:
                    # 最后一个元素可能是邮编
                    try:
                        postcode = state_postcode[-1]
                        # 检查是否为4位数字
                        if postcode.isdigit() and len(postcode) == 4:
                            # 州是邮编前的所有内容
                            state = ' '.join(state_postcode[:-1])
                        else:
                            postcode = None
                    except:
                        postcode = None
                
                # 倒数第二部分可能是郊区
                if len(address_parts) > 2:
                    suburb = address_parts[-2].strip()
                    
                # 第一部分可能是街道地址
                street_address = address_parts[0].strip()
        
        # 创建基本信息
        basic_info = PropertyBasicInfo(
            price_value=price_value,
            price_is_numeric=price_is_numeric,
            full_address=full_address,
            street_address=street_address,
            suburb=suburb,
            state=state,
            postcode=postcode,
            bedrooms_count=bedrooms_count,
            bathrooms_count=bathrooms_count,
            car_parks=response.car_parks,
            land_size=response.land_size,
            property_type=response.property_type,
        )
        
        # 创建事件信息
        events = PropertyEvents(
            inspection_date=response.inspection_date
        )
        
        # 创建媒体信息
        media = PropertyMedia(
            image_urls=response.image_urls,
            main_image_url=response.image_urls[0] if response.image_urls else None
        )
        
        # 创建代理信息
        agent = AgentInfo(
            agent_name=response.agent_name
        )
        
        return cls(
            listing_id=response.listing_id,
            basic_info=basic_info,
            media=media,
            agent=agent,
            events=events
        )
    
    def to_search_response(self) -> PropertySearchResponse:
        """Convert back to PropertySearchResponse"""
        # 将数值转换回字符串格式
        if self.basic_info.price_value:
            price_str = f"${self.basic_info.price_value:,.0f}"
        else:
            price_str = "Contact agent"
            
        bedrooms_str = str(self.basic_info.bedrooms_count) if self.basic_info.bedrooms_count is not None else "0"
        bathrooms_str = str(self.basic_info.bathrooms_count) if self.basic_info.bathrooms_count is not None else "0"
        
        return PropertySearchResponse(
            listing_id=self.listing_id,
            price=price_str,
            address=self.basic_info.full_address,
            bedrooms=bedrooms_str,
            bathrooms=bathrooms_str,
            car_parks=self.basic_info.car_parks,
            land_size=self.basic_info.land_size,
            property_type=self.basic_info.property_type,
            inspection_date=self.events.inspection_date,
            image_urls=self.media.image_urls,
            agent_name=self.agent.agent_name
        )
    
# Model for property recommendation
class PropertyRecommendationInfo(BaseModel):
    """Model for recommendation information"""
    score: float
    highlights: List[str]
    concerns: List[str]
    explanation: str
    
# Enhanced property response with recommendation information
class PropertyWithRecommendation(BaseModel):
    """Property with recommendation information"""
    property: FirestoreProperty
    recommendation: PropertyRecommendationInfo

# Response structure for recommendation endpoint
class PropertyRecommendationResponse(BaseModel):
    """Response model for property recommendations"""
    properties: List[PropertyWithRecommendation]

class ConversationMessage(BaseModel):
    """单条聊天消息模型，支持 user/assistant/tool 消息和类型、元数据"""
    role: Literal["user", "assistant", "tool", "preference_processor"] = Field(..., description="消息角色：'user'、'assistant' 或 'tool'")
    content: str = Field(..., description="消息内容")
    type: Optional[str] = Field(None, description="消息类型，如 'recommendation', 'search_result', 'tool' 等")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="额外元数据")
    timestamp: datetime = Field(default_factory=datetime.now)
    recommendation: Optional[PropertyRecommendationResponse] = Field(default=None, description="结构化推荐结果") 

class ChatSession(BaseModel):
    """聊天会话模型"""
    session_id: str = Field(..., description="会话唯一标识符")
    messages: List[ConversationMessage] = Field(default_factory=list, description="会话中的所有消息")
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    preferences: Optional[Dict] = Field(default=None, description="用户偏好")
    search_params: Optional[Dict] = Field(default=None, description="搜索参数")

class SavedProperties(BaseModel):
    """Model for storing saved properties for a session"""
    session_id: str
    properties: List[PropertyWithRecommendation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
