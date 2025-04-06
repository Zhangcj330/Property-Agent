from backend.app.services.planning_service import get_planning_info
import json
import asyncio

# 测试一些不同类型的地址
test_addresses = [
    "123 Pitt Street, Sydney NSW 2000",  # 市中心
    "1 Notts Avenue, Bondi Beach NSW 2026",  # 海滩区
    "231 Miller Street, North Sydney NSW 2060",  # 北悉尼
    "180 George Street, Parramatta NSW 2150",  # 帕拉马塔
]

def print_planning_info(result):
    if result.error:
        print(f"Error: {result.error}")
        return
        
    summary = result.get_summary()
    print("\n=== Planning Information ===")
    print(f"Address: {summary['address']}")
    print(f"Zone: {summary['zone_name']}")
    print(f"Height Limit: {summary['height_limit']}m")
    print(f"Floor Space Ratio: {summary['floor_space_ratio']}")
    print(f"Heritage: {'Yes' if summary['is_heritage'] else 'No'}")
    print("\n=== Hazard Information ===")
    print(f"Flood Risk: {'Yes' if summary['flood_risk'] else 'No'}")
    print(f"Landslide Risk: {'Yes' if summary['landslide_risk'] else 'No'}")
    print("\n" + "="*30 + "\n")

async def process_address(address):
    print(f"\nProcessing: {address}")
    result = await get_planning_info(address)
    print_planning_info(result)

async def main():
    tasks = [process_address(address) for address in test_addresses]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main()) 