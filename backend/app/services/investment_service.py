import duckdb
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class InvestmentMetrics:
    """Core investment metrics for a property"""
    suburb: str
    rental_yield: Optional[float] = None  # Annual rental yield as percentage
    capital_gain: Optional[float] = None  # 1-year capital gain as percentage
    current_price: Optional[float] = None  # Current median price
    weekly_rent: Optional[float] = None   # Current weekly rent


class InvestmentService:
    def __init__(self):
        """Initialize the investment service with DuckDB database connection"""
        # Use project root as base, matching sql_service.py
        root_dir = Path(__file__).parent.parent.parent
        db_path = root_dir / "db" / "suburb_data.duckdb"
        self.db_path = str(db_path)
        
        # Test connection to ensure database is accessible
        self._test_connection()

    @contextmanager
    def _get_connection(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Get a database connection with proper error handling and automatic cleanup"""
        import time
        conn = None
        last_error = None
        for attempt in range(max_retries):
            try:
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

    def _test_connection(self):
        """Test database connection and verify table exists"""
        try:
            with self._get_connection() as conn:
                # Verify suburbs table exists
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = [table[0] for table in tables]
                if 'suburbs' not in table_names:
                    raise Exception(f"Required 'suburbs' table not found in database. Available tables: {table_names}")
        except Exception as e:
            raise Exception(f"Failed to connect to database at {self.db_path}: {str(e)}")

    def get_investment_metrics(self, suburb: str) -> InvestmentMetrics:
        """Get core investment metrics for a suburb from DuckDB
        
        Args:
            suburb: Name of the suburb
            
        Returns:
            InvestmentMetrics containing rental yield and capital gain data from DuckDB
        """
        try:
            metrics = InvestmentMetrics(suburb=suburb)
            
            with self._get_connection() as conn:
                # Query suburb data with case-insensitive search
                query = """
                SELECT 
                    Suburb_Name,
                    Smart_Median_House_Price,
                    Gross_Yield,
                    Annualized_growth_Forecast_next_four_years,
                    Rent
                FROM suburbs 
                WHERE LOWER(Suburb_Name) = LOWER(?)
                LIMIT 1
                """
                
                result = conn.execute(query, [suburb]).fetchone()
                
                if result:
                    suburb_name, median_price, gross_yield, forecast_growth, rent = result
                    
                    # Map database fields to InvestmentMetrics
                    if median_price is not None:
                        metrics.current_price = float(median_price)
                    
                    if gross_yield is not None:
                        # Convert from decimal to percentage
                        metrics.rental_yield = round(gross_yield * 100, 2)
                    
                    if forecast_growth is not None:
                        # Convert from decimal to percentage  
                        metrics.capital_gain = round(forecast_growth * 100, 2)
                    
                    if rent is not None:
                        metrics.weekly_rent = float(rent)
                
                else:
                    print(f"No data found for suburb: {suburb}")
            
            return metrics
            
        except Exception as e:
            print(f"Error getting investment metrics for {suburb}: {str(e)}")
            return InvestmentMetrics(suburb=suburb)
    
