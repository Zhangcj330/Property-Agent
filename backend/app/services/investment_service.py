import pandas as pd
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

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
        """Initialize the investment service with rental and price data"""
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent / "data"
        
        # Load rental data
        rental_file = data_dir / "rental_prices.csv"
        self._rental_data = pd.read_csv(rental_file)
        
        # Load suburb price data
        price_file = data_dir / "suburb_average_price_yearly.csv"
        self._price_data = pd.read_csv(price_file)
        
        # Clean and prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Clean and prepare the data for analysis"""
        # Convert price data to long format for easier analysis
        self._price_data = pd.melt(
            self._price_data,
            id_vars=['Property locality'],
            value_vars=['2023', '2024'],  # Only need last 2 years for capital gain
            var_name='year',
            value_name='median_price'
        )
        self._price_data = self._price_data.rename(columns={'Property locality': 'suburb'})
        self._price_data['year'] = pd.to_numeric(self._price_data['year'])
        
        # Sort by suburb and year in descending order (most recent first)
        self._price_data = self._price_data.sort_values(['suburb', 'year'], ascending=[True, False])
        
        # Clean rental data - ensure numeric columns are float
        self._rental_data['Postcode'] = pd.to_numeric(self._rental_data['Postcode'], errors='coerce')
        self._rental_data['Bedrooms'] = pd.to_numeric(self._rental_data['Bedrooms'], errors='coerce')
        for year in ['2021', '2022', '2023', '2024']:
            self._rental_data[year] = pd.to_numeric(self._rental_data[year], errors='coerce')
        
        # Drop rows with invalid data
        self._rental_data = self._rental_data.dropna(subset=['Postcode', 'Bedrooms', '2024'])
    
    def get_investment_metrics(self, suburb: str, postcode: str, bedrooms: int = 2) -> InvestmentMetrics:
        """Get core investment metrics for a suburb
        
        Args:
            suburb: Name of the suburb
            postcode: Postcode of the suburb
            bedrooms: Number of bedrooms for rental calculation (default 2)
            
        Returns:
            InvestmentMetrics containing rental yield and capital gain data
        """
        try:
            # Initialize result
            metrics = InvestmentMetrics(suburb=suburb)
            
            # Get price data for the suburb
            suburb_data = self._price_data[self._price_data['suburb'].str.lower() == suburb.lower()]
            suburb_data = suburb_data.dropna(subset=['median_price'])
            
            if not suburb_data.empty:
                # Get latest price
                latest_data = suburb_data.iloc[0]
                metrics.current_price = latest_data['median_price']
                
                # Calculate capital gain
                if len(suburb_data) > 1:
                    previous_price = suburb_data.iloc[1]['median_price']
                    if previous_price > 0:  # Avoid division by zero
                        metrics.capital_gain = round(
                            ((metrics.current_price / previous_price) - 1) * 100, 
                            2
                        )
            
            # Get rental data - convert postcode to float for comparison
            try:
                postcode_float = float(postcode)
                rental_row = self._rental_data[
                    (self._rental_data['Postcode'] == postcode_float) &
                    (self._rental_data['Bedrooms'] == float(bedrooms))
                ]
                
                if not rental_row.empty:
                    # Get the latest weekly rent
                    latest_rent = rental_row.iloc[0]['2024']
                    if pd.notna(latest_rent):
                        metrics.weekly_rent = latest_rent
                    
                    # Calculate rental yield if we have both price and rent
                    if metrics.weekly_rent and metrics.current_price and metrics.current_price > 0:
                        annual_rent = metrics.weekly_rent * 52
                        metrics.rental_yield = round(
                            (annual_rent / metrics.current_price) * 100,
                            2
                        )
            except ValueError as e:
                print(f"Warning: Invalid postcode format: {postcode}")
                print(f"Error details: {str(e)}")
            
            return metrics
            
        except Exception as e:
            print(f"Error getting investment metrics for {suburb}: {str(e)}")
            return InvestmentMetrics(suburb=suburb)
    
    def get_top_rental_yield_suburbs(self, min_price: float = 0, max_price: float = float('inf'), 
                                   bedrooms: int = 2, limit: int = 10) -> List[InvestmentMetrics]:
        """Get suburbs with the highest rental yields within a price range
        
        Args:
            min_price: Minimum property price to consider
            max_price: Maximum property price to consider
            bedrooms: Number of bedrooms for rental calculation
            limit: Maximum number of results to return
            
        Returns:
            List of InvestmentMetrics sorted by rental yield (descending)
        """
        results = []
        
        # Get unique suburbs with 2024 prices
        suburb_prices = self._price_data[
            (self._price_data['year'] == 2024) & 
            (self._price_data['median_price'].notna()) &
            (self._price_data['median_price'] >= min_price) &
            (self._price_data['median_price'] <= max_price)
        ]
        
        # Calculate metrics for each suburb
        for _, row in suburb_prices.iterrows():
            metrics = self.get_investment_metrics(row['suburb'], "2000", bedrooms)
            if metrics.rental_yield:
                results.append(metrics)
        
        # Sort by rental yield and return top results
        return sorted(results, key=lambda x: x.rental_yield or 0, reverse=True)[:limit]
    
    def get_top_capital_gain_suburbs(self, min_price: float = 0, max_price: float = float('inf'),
                                   limit: int = 10) -> List[InvestmentMetrics]:
        """Get suburbs with the highest capital gains within a price range
        
        Args:
            min_price: Minimum property price to consider
            max_price: Maximum property price to consider
            limit: Maximum number of results to return
            
        Returns:
            List of InvestmentMetrics sorted by capital gain (descending)
        """
        results = []
        
        # Get unique suburbs with 2024 prices
        suburb_prices = self._price_data[
            (self._price_data['year'] == 2024) & 
            (self._price_data['median_price'].notna()) &
            (self._price_data['median_price'] >= min_price) &
            (self._price_data['median_price'] <= max_price)
        ]
        
        # Calculate metrics for each suburb
        for _, row in suburb_prices.iterrows():
            metrics = self.get_investment_metrics(row['suburb'], "2000", 2)
            if metrics.capital_gain:
                results.append(metrics)
        
        # Sort by capital gain and return top results
        return sorted(results, key=lambda x: x.capital_gain or 0, reverse=True)[:limit]
