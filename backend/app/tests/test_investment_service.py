import pytest
from app.services.investment_service import InvestmentService

def test_print_chatswood_data():
    """Print investment metrics for Chatswood to understand the data"""
    service = InvestmentService()
    
    # Get metrics for different bedroom counts
    metrics_2bed = service.get_investment_metrics("Chatswood", postcode="2067", bedrooms=2)
    metrics_3bed = service.get_investment_metrics("Chatswood", postcode="2067", bedrooms=3)
    metrics_4bed = service.get_investment_metrics("Chatswood", postcode="2067", bedrooms=4)
    
    print("\nChatswood Investment Metrics:")
    print(f"\n2 Bedrooms:")
    print(f"Current Price: ${metrics_2bed.current_price:,.2f}" if metrics_2bed.current_price else "Current Price: N/A")
    print(f"Weekly Rent: ${metrics_2bed.weekly_rent:.2f}" if metrics_2bed.weekly_rent else "Weekly Rent: N/A")
    print(f"Rental Yield: {metrics_2bed.rental_yield:.2f}%" if metrics_2bed.rental_yield else "Rental Yield: N/A")
    print(f"Capital Gain: {metrics_2bed.capital_gain:.2f}%" if metrics_2bed.capital_gain else "Capital Gain: N/A")
    
    print(f"\n3 Bedrooms:")
    print(f"Current Price: ${metrics_3bed.current_price:,.2f}" if metrics_3bed.current_price else "Current Price: N/A")
    print(f"Weekly Rent: ${metrics_3bed.weekly_rent:.2f}" if metrics_3bed.weekly_rent else "Weekly Rent: N/A")
    print(f"Rental Yield: {metrics_3bed.rental_yield:.2f}%" if metrics_3bed.rental_yield else "Rental Yield: N/A")
    print(f"Capital Gain: {metrics_3bed.capital_gain:.2f}%" if metrics_3bed.capital_gain else "Capital Gain: N/A")
    
    print(f"\n4 Bedrooms:")
    print(f"Current Price: ${metrics_4bed.current_price:,.2f}" if metrics_4bed.current_price else "Current Price: N/A")
    print(f"Weekly Rent: ${metrics_4bed.weekly_rent:.2f}" if metrics_4bed.weekly_rent else "Weekly Rent: N/A")
    print(f"Rental Yield: {metrics_4bed.rental_yield:.2f}%" if metrics_4bed.rental_yield else "Rental Yield: N/A")
    print(f"Capital Gain: {metrics_4bed.capital_gain:.2f}%" if metrics_4bed.capital_gain else "Capital Gain: N/A")
