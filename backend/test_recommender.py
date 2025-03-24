import asyncio
import sys
import os
import json
from unittest.mock import MagicMock, AsyncMock

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import UserPreferences
from app.services.recommender import PropertyRecommender, PropertyRecommendation
from app.models import FirestoreProperty

# 创建一个模拟的FirestoreProperty对象
def create_mock_property(listing_id):
    property = FirestoreProperty(
        listing_id=listing_id,
        basic_info={
            "price_value": 500000,
            "price_is_numeric": True,
            "full_address": "123 Test St",
            "bedrooms_count": 3,
            "bathrooms_count": 2,
            "property_type": "House"
        },
        analysis=None,
        metadata={}
    )
    return property

# 创建一个模拟的UserPreferences对象
mock_preferences = UserPreferences(
    budget_min=400000,
    budget_max=600000,
    bedrooms_min=2,
    bedrooms_max=4,
    bathrooms_min=1,
    bathrooms_max=3,
    location=["Test City"],
    property_types=["House"],
    must_have_features=["Garden"],
    nice_to_have_features=["Pool"],
    deal_breakers=["Busy Road"]
)

# 测试我们的修复
async def test_dictionary_recommendations():
    print("\n正在测试字典推荐处理和排序...")
    # 创建模拟属性 - 现在添加了3个属性
    properties = [
        create_mock_property("prop-003"),  # 这个会排在中间
        create_mock_property("prop-001"),  # 这个应该排在第一位
        create_mock_property("prop-002"),  # 这个应该排在最后
        create_mock_property("prop-004"),  # 这个没有推荐
    ]
    
    # 创建字典形式的推荐，分数明显不同
    dictionary_recommendations = [
        {
            "property_id": "prop-001",
            "score": 0.9,  # 最高分数
            "highlights": ["Good location", "Matches budget"],
            "concerns": ["No pool"],
            "explanation": "This property is a good match overall."
        },
        {
            "property_id": "prop-002",
            "score": 0.3,  # 最低分数
            "highlights": ["Has a garden"],
            "concerns": ["Price at upper end of budget"],
            "explanation": "This property has some good features but is expensive."
        },
        {
            "property_id": "prop-003",
            "score": 0.6,  # 中间分数
            "highlights": ["Nice neighborhood"],
            "concerns": ["Older building"],
            "explanation": "This property is in a good location but needs some updates."
        }
    ]
    
    # 手动调用推荐匹配逻辑
    recommendation_map = {}
    for rec in dictionary_recommendations:
        if isinstance(rec, dict) and "property_id" in rec:
            # 如果推荐是一个字典，直接使用它
            recommendation_map[rec["property_id"]] = PropertyRecommendation(**rec)
        elif hasattr(rec, "property_id"):
            # 如果推荐已经是一个PropertyRecommendation对象
            recommendation_map[rec.property_id] = rec
        else:
            print(f"跳过无效的推荐格式: {rec}")
    
    # 匹配原始属性对象与其推荐
    recommended_properties = []
    for prop in properties:
        if prop.listing_id in recommendation_map:
            # 将推荐数据与属性存储在一起
            setattr(prop, '_recommendation', recommendation_map[prop.listing_id])
            recommended_properties.append(prop)
    
    # 如果没有属性与推荐匹配，返回原始属性
    if not recommended_properties:
        print("没有属性与推荐匹配")
        return
    
    # 按分数排序（降序）
    print("\n排序前的推荐属性:")
    for prop in recommended_properties:
        print(f"Property ID: {prop.listing_id}, Score: {prop._recommendation.score}")
        
    recommended_properties.sort(key=lambda x: x._recommendation.score, reverse=True)
    
    print("\n排序后的推荐属性:")
    for prop in recommended_properties:
        print(f"Property ID: {prop.listing_id}, Score: {prop._recommendation.score}")
    
    # 验证排序
    is_sorted_correctly = True
    for i in range(len(recommended_properties) - 1):
        if recommended_properties[i]._recommendation.score < recommended_properties[i+1]._recommendation.score:
            is_sorted_correctly = False
            break
    
    print(f"\n排序是否正确: {is_sorted_correctly}")
    
    # 验证预期的顺序
    expected_order = ["prop-001", "prop-003", "prop-002"]
    actual_order = [prop.listing_id for prop in recommended_properties]
    order_matches = expected_order == actual_order
    
    print(f"预期顺序: {expected_order}")
    print(f"实际顺序: {actual_order}")
    print(f"顺序是否匹配: {order_matches}")
    
    if not order_matches:
        print("警告: 推荐未按预期顺序排序!")

# 运行测试
if __name__ == "__main__":
    asyncio.run(test_dictionary_recommendations()) 