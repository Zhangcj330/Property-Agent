import sys
import os
import asyncio

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 现在可以导入app模块了
from app.llm_service import LLMService
from test_preference_extraction import TestPreferenceExtraction

async def run_test(test_name):
    """运行单个测试场景"""
    test_class = TestPreferenceExtraction()
    test_method = getattr(test_class, test_name)
    
    # 创建LLM服务实例
    llm_service = test_class.llm_service()
    
    # 运行测试
    await test_method(llm_service)
    print(f"\n测试完成: {test_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("可用的测试场景:")
        print("  test_clear_preferences_extraction - 测试明确偏好提取")
        print("  test_vague_preferences_extraction - 测试模糊偏好提取")
        print("  test_contradictory_preferences_extraction - 测试矛盾偏好提取")
        print("  test_changing_preferences_extraction - 测试变化偏好提取")
        print("  test_minimal_information_extraction - 测试最小信息提取")
        print("\n请指定要运行的测试名称，例如:")
        print("python run_test.py test_clear_preferences_extraction")
        sys.exit(1)
        
    test_name = sys.argv[1]
    asyncio.run(run_test(test_name)) 