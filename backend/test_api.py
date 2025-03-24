import requests
import json
from datetime import datetime

# API端点
url = "http://localhost:8000/api/v1/recommend"

# 获取当前时间
now = datetime.now().isoformat()

# 创建测试属性 - 更完整的结构
test_properties = [
    {
        "listing_id": "prop-001",
        "basic_info": {
            "price_value": 500000,
            "price_is_numeric": True,
            "full_address": "123 Test St, Test City",
            "street_address": "123 Test St",
            "suburb": "Test City",
            "state": "Test State",
            "postcode": "1234",
            "bedrooms_count": 3,
            "bathrooms_count": 2,
            "car_parks": "2",
            "land_size": "500 sqm",
            "property_type": "House"
        },
        "media": {
            "image_urls": ["http://example.com/image1.jpg", "http://example.com/image2.jpg"],
            "main_image_url": "http://example.com/image1.jpg",
            "video_url": None
        },
        "agent": {
            "agent_name": "Test Agent",
            "agency": "Test Agency",
            "contact_number": "0412345678",
            "email": "agent@test.com"
        },
        "events": {
            "inspection_date": None,
            "inspection_times": None,
            "auction_date": None,
            "listing_date": None,
            "last_updated_date": None
        },
        "metadata": {
            "created_at": now,
            "updated_at": now,
            "last_analysis_at": None,
            "source": "test",
            "status": "active"
        },
        "analysis": None
    },
    {
        "listing_id": "prop-002",
        "basic_info": {
            "price_value": 450000,
            "price_is_numeric": True,
            "full_address": "456 Test Ave, Test City",
            "street_address": "456 Test Ave",
            "suburb": "Test City",
            "state": "Test State",
            "postcode": "1234",
            "bedrooms_count": 2,
            "bathrooms_count": 1,
            "car_parks": "1",
            "land_size": None,
            "property_type": "Apartment"
        },
        "media": {
            "image_urls": ["http://example.com/image3.jpg"],
            "main_image_url": "http://example.com/image3.jpg",
            "video_url": None
        },
        "agent": {
            "agent_name": "Test Agent 2",
            "agency": "Test Agency 2",
            "contact_number": "0412345679",
            "email": "agent2@test.com"
        },
        "events": {
            "inspection_date": None,
            "inspection_times": None,
            "auction_date": None,
            "listing_date": None,
            "last_updated_date": None
        },
        "metadata": {
            "created_at": now,
            "updated_at": now,
            "last_analysis_at": None,
            "source": "test",
            "status": "active"
        },
        "analysis": None
    }
]

# 创建测试用户偏好
test_preferences = {
    "Features": {"preference": "modern", "confidence_score": 0.8, "weight": 0.8},
    "Layout": {"preference": "open plan", "confidence_score": 0.7, "weight": 0.7},
    "Condition": {"preference": "new", "confidence_score": 0.9, "weight": 0.9},
    "Environment": {"preference": "quiet", "confidence_score": 0.8, "weight": 0.8},
    "Style": {"preference": "contemporary", "confidence_score": 0.7, "weight": 0.7},
    "Quality": {"preference": "high", "confidence_score": 0.9, "weight": 0.9},
    "SchoolDistrict": {"preference": "good", "confidence_score": 0.8, "weight": 0.8},
    "Community": {"preference": "family friendly", "confidence_score": 0.7, "weight": 0.7},
    "Transport": {"preference": "close to public transport", "confidence_score": 0.6, "weight": 0.6},
    "Other": {"preference": "none", "confidence_score": 0.5, "weight": 0.5}
}

# 创建请求体
request_body = {
    "properties": test_properties,
    "preferences": test_preferences
}

# 打印请求数据 (调试)
print("正在发送请求到:", url)
print("请求头信息:", {"Content-Type": "application/json"})
print(f"请求主体(部分): {len(test_properties)}个属性, 带有完整的偏好")

# 发送请求，并捕获异常
try:
    response = requests.post(url, json=request_body)
    # 打印响应状态码
    print(f"状态码: {response.status_code}")

    # 打印响应内容
    if response.status_code == 200:
        try:
            data = response.json()
            
            # 漂亮打印JSON
            print("\n响应数据结构:")
            print(json.dumps(data, indent=2))
            
            # 打印推荐属性详情
            print("\n推荐属性详情:")
            if "properties" in data and data["properties"]:
                for idx, property_rec in enumerate(data["properties"]):
                    prop = property_rec["property"]
                    rec = property_rec["recommendation"]
                    print(f"\n推荐 #{idx+1}: {prop['listing_id']}")
                    print(f"  分数: {rec['score']}")
                    print(f"  优点: {', '.join(rec['highlights'])}")
                    print(f"  缺点: {', '.join(rec['concerns'])}")
                    print(f"  解释: {rec['explanation']}")
            else:
                print("没有返回推荐属性")
        except json.JSONDecodeError:
            print("响应不是有效的JSON格式:", response.text[:200])
    else:
        try:
            error_data = response.json()
            print(f"错误响应: {json.dumps(error_data, indent=2)}")
        except:
            print(f"错误: {response.text[:200]}")
except Exception as e:
    print(f"请求异常: {str(e)}") 