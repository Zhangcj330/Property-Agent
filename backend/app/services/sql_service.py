import duckdb
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import settings
from pathlib import Path
import time
from contextlib import contextmanager

class SQLQueryRequest(BaseModel):
    """Request model for SQL query generation"""
    user_query: str
    table_schema: Optional[str] = None
    filters: Optional[list[str]] = None
    
class SQLQueryResponse(BaseModel):
    """Response model for SQL query execution"""
    query: str
    results: List[Dict[str, Any]]
    error: Optional[str] = None

class SQLService:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # 使用项目根目录作为基准来构建数据库路径
            root_dir = Path(__file__).parent.parent.parent
            db_path = str(root_dir / "db" / "suburb_data.duckdb")
        
        self.db_path = db_path
        self.llm = ChatGoogleGenerativeAI(
            api_key=settings.GEMINI_API_KEY,
            model="gemini-2.0-flash",
        )
        
        # 确保数据库文件存在
        db_dir = Path(db_path).parent
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache table schema on initialization
        with self._get_connection() as conn:
            self.table_schema = self._get_table_schema(conn)
    
    @contextmanager
    def _get_connection(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Get a database connection with proper error handling and automatic cleanup"""
        conn = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # 使用最简单的只读模式
                conn = duckdb.connect(self.db_path, read_only=True)
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        if not conn:
            raise Exception(f"Failed to connect to database after {max_retries} attempts: {str(last_error)}")
        
        try:
            yield conn
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def _get_table_schema(self, conn) -> str:
        """Get the schema of all tables in the database"""
        try:
            # Get list of tables
            tables = conn.execute(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
                """
            ).fetchall()
            
            schema = []
            for table in tables:
                table_name = table[0]
                # Get column information for each table
                columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
                schema.append(f"Table: {table_name}")
                schema.append("Columns:")
                for col in columns:
                    schema.append(f"  - {col[0]} ({col[1]})")
                schema.append("")
            
            return "\n".join(schema)
        except Exception as e:
            print(f"Error getting schema: {e}")
            return ""

    async def generate_sql_query(self, request: SQLQueryRequest) -> str:
        """Generate SQL query based on user question using LLM"""
        # Get sample data
        with self._get_connection() as conn:
            sample_data = []
            tables = conn.execute(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
                """
            ).fetchall()
            
            for table in tables:
                table_name = table[0]
                # Get top 10 rows from each table
                try:
                    rows = conn.execute(f"SELECT * FROM {table_name} LIMIT 10").fetchall()
                    if rows:
                        columns = [desc[0] for desc in conn.description]
                        sample_data.append(f"\nSample data from {table_name}:")
                        # Add column headers
                        sample_data.append("| " + " | ".join(columns) + " |")
                        sample_data.append("|" + "|".join(["-" * len(col) for col in columns]) + "|")
                        # Add row data
                        for row in rows:
                            sample_data.append("| " + " | ".join(str(val) for val in row) + " |")
                except Exception as e:
                    print(f"Error getting sample data from {table_name}: {e}")
                    continue

        context = f"""
You are an expert SQL query generator, specializing in real estate data. Your task is to convert natural language queries into SQL queries for a DuckDB database to **search and retrieve detailed suburb-level property information**.
In addition to SQL syntax, you must also reason about geographic references in the user's query.

Database Schema:
{request.table_schema or self.table_schema}

Sample Data:
{"".join(sample_data)}

User Intent:
{request.user_query or ''}

Additional Context:
{request.filters or ''}

Your Responsibilities:
1. Understand the user's geographic or property-related search intent.
2. Retrieve **suburb-level property insights**, such as:
   - suburb_name
   - state
   - median_price
   - rental_yield
   - annual_growth
   - number_of_listings
   - demographics (if available)
3. Add computed fields where helpful (e.g., price per bedroom, yield bands).
4. Use meaningful column aliases for any computed values.
5. Use appropriate filters, sorting, or grouping based on user needs.
6. Limit results to a reasonable number (e.g., LIMIT 10) unless specified otherwise.
7. Apply multiple sorting criteria to recommend suburb that best match the user's Intent.

Rules:
- Generate **only** the raw SQL query (no explanations, markdown, or comments).
- Use correct DuckDB syntax and apply table aliases.
- Reference sample data to ensure correct value formats, column names, and data types.

SQL Query:
"""

        response = self.llm.invoke([
            SystemMessage(content=context),
            HumanMessage(content="Write the SQL query only, no markdown formatting, no explanations or additional text.")
        ])

        # Clean up the response - remove any markdown formatting
        query = response.content.strip()
        if query.startswith("```sql"):
            query = query[6:]
        if query.startswith("```"):
            query = query[3:]
        if query.endswith("```"):
            query = query[:-3]
        
        return query.strip()

    async def execute_query(self, query: str) -> SQLQueryResponse:
        """Execute the SQL query and return results"""
        try:
            with self._get_connection() as conn:
                # Execute query
                result = conn.execute(query)
                
                # Get column names
                column_names = [desc[0] for desc in result.description]
                
                # Fetch results and convert to list of dicts
                rows = result.fetchall()
                results = []
                for row in rows:
                    result_dict = {}
                    for i, value in enumerate(row):
                        # Convert any special types to string if needed
                        if isinstance(value, (duckdb.Value, bytes)):
                            value = str(value)
                        result_dict[column_names[i]] = value
                    results.append(result_dict)

                return SQLQueryResponse(
                    query=query,
                    results=results,
                )
                
        except Exception as e:
            return SQLQueryResponse(
                query=query,
                results=[],
                error=str(e)
            )

    async def process_question(self, query: str, filters: str = None) -> SQLQueryResponse:
        """Process a natural language question and return query results"""
        # Generate query
        query = await self.generate_sql_query(SQLQueryRequest(
            user_query=query,
            filters=filters
        ))
        
        # Execute query
        return await self.execute_query(query) 