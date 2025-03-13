# Basic usage example
from fvg import FVGStrategy

# Create strategy with Polygon.io API key
strategy = FVGStrategy(
    symbol='QQQ',
    risk_percent=1.0,
    rr_ratio=1.2,
    start_time='09:31',
    end_time='10:00',
    polygon_api_key='pO9FtYtmGMqrS16e7Bw6Prw34ZVGjw5p'
)

# Fetch data from Polygon.io
data = strategy.fetch_polygon_data(
    start_date='2024-01-01',
    end_date='2024-01-31'
)

# Run backtest
results = strategy.backtest(data)

# Display and save results
strategy.plot_results(results)