import asyncio
from pathlib import Path
import sys

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent.parent
sys.path.append(str(backend_dir))

from app.services.sql_service import SQLService

async def test_connection():
    # Get the absolute path to the database
    db_path = backend_dir / "db" / "suburb_data.duckdb"
    print(f"Database path: {db_path}")
    print(f"Database exists: {db_path.exists()}")
    
    # Initialize service
    sql_service = SQLService(str(db_path))
    
    # Test schema retrieval
    print("\nTesting schema retrieval:")
    print(sql_service.table_schema)
    
    # Test simple query
    print("\nTesting simple query:")
    response = await sql_service.process_question("Show me the top 5 suburbs with the highest rental yield")
    print(f"Query: {response.query}")
    print(f"Results: {response.results}")
    print(f"Error: {response.error}")

if __name__ == "__main__":
    asyncio.run(test_connection()) 