import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, time, timedelta
import pytz
from pandas.plotting import register_matplotlib_converters
import requests
from urllib.parse import urlencode

register_matplotlib_converters()

class FVGStrategy:
    def __init__(self, symbol='QQQ', risk_percent=1.0, rr_ratio=1.2, 
                 start_time='09:31', end_time='10:00', timezone='US/Eastern',
                 fvg_lookback=20, polygon_api_key=None):
        """
        Initialize the FVG trading strategy
        
        Parameters:
        ----------
        symbol : str
            Trading symbol (default: 'QQQ' for NASDAQ-100 ETF)
        risk_percent : float
            Risk percentage per trade (default: 1.0%)
        rr_ratio : float
            Risk-reward ratio for take profit (default: 1.2)
        start_time : str
            Start time for trading window in NY time (default: '09:31')
        end_time : str
            End time for trading window in NY time (default: '10:00')
        timezone : str
            Timezone for trading window (default: 'US/Eastern')
        fvg_lookback : int
            Number of bars to look back for FVGs (default: 20)
        polygon_api_key : str, optional
            API key for Polygon.io data (default: None)
        """
        self.symbol = symbol
        self.risk_percent = risk_percent
        self.rr_ratio = rr_ratio
        self.start_time = pd.to_datetime(start_time).time()
        self.end_time = pd.to_datetime(end_time).time()
        self.timezone = pytz.timezone(timezone)
        self.fvg_lookback = fvg_lookback
        self.polygon_api_key = polygon_api_key
        self.initial_balance = 100000  # Starting with $100,000
        self.current_balance = self.initial_balance
        self.positions = []
        self.trade_history = []
    
    def fetch_polygon_data(self, start_date, end_date, interval='1'):
        """
        Fetch historical data from Polygon.io
        
        Parameters:
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        interval : str
            Data interval in minutes (default: '1')
            
        Returns:
        -------
        DataFrame
            Historical price data
        """
        if not self.polygon_api_key:
            raise ValueError("Polygon API key is required for fetching data from Polygon.io")
        
        # Convert dates to Unix milliseconds
        start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)
        
        print(f"Fetching {self.symbol} data from Polygon.io...")
        
        # Construct the URL
        base_url = "https://api.polygon.io/v2/aggs/ticker"
        ticker = self.symbol.upper()
        multiplier = interval
        timespan = "minute"
        
        # Format dates for the API
        from_date = start_date
        to_date = end_date
        
        # Prepare query parameters
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.polygon_api_key
        }
        
        # Construct final URL
        url = f"{base_url}/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}?{urlencode(params)}"
        
        # Make the request
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
        
        data = response.json()
        
        if not data['results']:
            raise ValueError(f"No data available for {self.symbol} in the specified date range")
        
        # Convert to DataFrame
        df = pd.DataFrame(data['results'])
        
        # Rename columns to match Yahoo Finance format
        df = df.rename(columns={
            'v': 'Volume',
            'o': 'Open',
            'c': 'Close',
            'h': 'High',
            'l': 'Low',
            't': 'timestamp'
        })
        
        # Convert timestamp from milliseconds to datetime
        df['Datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timezone
        df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert(self.timezone)
        
        # Set as index
        df = df.set_index('Datetime')
        
        # Filter only to trading hours
        trading_hours = df.between_time('09:30', '16:00')
        
        print(f"Fetched {len(trading_hours)} data points")
        return trading_hours
    
    def download_data(self, start_date, end_date, interval='1m'):
        """
        Download historical data from Yahoo Finance
        
        Parameters:
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        interval : str
            Data interval (default: '1m')
            
        Returns:
        -------
        DataFrame
            Historical price data
        """
        # Add a buffer day before and after to ensure we have all trading hours
        start = pd.to_datetime(start_date) - timedelta(days=1)
        end = pd.to_datetime(end_date) + timedelta(days=1)
        
        print(f"Downloading {self.symbol} data from {start_date} to {end_date}...")
        df = yf.download(self.symbol, start=start, end=end, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data available for {self.symbol} in the specified date range")
        
        # Reset index to make datetime a column and localize timezone
        df = df.reset_index()
        df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize('UTC').dt.tz_convert(self.timezone)
        df = df.set_index('Datetime')
        
        # Filter only to trading hours
        trading_hours = df.between_time('09:30', '16:00')
        
        print(f"Downloaded {len(trading_hours)} data points")
        return trading_hours
    # [Include the rest of the FVGStrategy methods from the previous implementation]
    
    def is_trade_time(self, timestamp):
        """Check if current time is within trading window"""
        time_of_day = timestamp.time()
        return time_of_day >= self.start_time and time_of_day < self.end_time
    
    def check_liquidity_sweep(self, df, current_idx):
        """Check for liquidity sweep (price breaking a swing point and reversing)"""
        if current_idx < 30:
            return False
        
        lookback_df = df.iloc[current_idx-30:current_idx+1]
        
        # Find swing highs and lows
        for i in range(5, 20):
            idx = -i - 1  # Negative indexing to get from the end
            
            # Check for swing high
            if (lookback_df['High'].iloc[idx] > lookback_df['High'].iloc[idx-1] and 
                lookback_df['High'].iloc[idx] > lookback_df['High'].iloc[idx+1] and
                lookback_df['High'].iloc[idx] > lookback_df['High'].iloc[idx-2] and
                lookback_df['High'].iloc[idx] > lookback_df['High'].iloc[idx+2]):
                
                # Check if price swept above this high and then reversed
                if (lookback_df['High'].iloc[-2] > lookback_df['High'].iloc[idx] and 
                    lookback_df['Close'].iloc[-2] < lookback_df['High'].iloc[idx]):
                    return True
            
            # Check for swing low
            if (lookback_df['Low'].iloc[idx] < lookback_df['Low'].iloc[idx-1] and 
                lookback_df['Low'].iloc[idx] < lookback_df['Low'].iloc[idx+1] and
                lookback_df['Low'].iloc[idx] < lookback_df['Low'].iloc[idx-2] and
                lookback_df['Low'].iloc[idx] < lookback_df['Low'].iloc[idx+2]):
                
                # Check if price swept below this low and then reversed
                if (lookback_df['Low'].iloc[-2] < lookback_df['Low'].iloc[idx] and 
                    lookback_df['Close'].iloc[-2] > lookback_df['Low'].iloc[idx]):
                    return True
        
        return False
    
    def check_strong_movement(self, df, current_idx):
        """Check for strong price movement"""
        if current_idx < 6:
            return False
        
        lookback_df = df.iloc[current_idx-6:current_idx]
        
        # Calculate ranges of last 2 and last 4 candles
        last_2_range = (lookback_df['High'].iloc[-2:] - lookback_df['Low'].iloc[-2:]).sum()
        last_4_range = (lookback_df['High'].iloc[-4:] - lookback_df['Low'].iloc[-4:]).sum()
        
        # Check if last 2 candles have more range than half of last 4 candles
        return last_2_range > (last_4_range / 2)
    
    def find_bullish_fvg(self, df, current_idx):
        """Find bullish Fair Value Gap"""
        if current_idx < self.fvg_lookback:
            return -1
        
        lookback_df = df.iloc[current_idx-self.fvg_lookback:current_idx+1]
        
        # Look for a bullish FVG (3-candle formation)
        for i in range(2, self.fvg_lookback - 1):
            idx = -i - 1  # Convert to negative indexing from the end
            
            # Bullish FVG: Low of the third candle is higher than the high of the first candle
            if lookback_df['Low'].iloc[idx] > lookback_df['High'].iloc[idx+2]:
                return idx
        
        return -1
    
    def find_bearish_fvg(self, df, current_idx):
        """Find bearish Fair Value Gap"""
        if current_idx < self.fvg_lookback:
            return -1
        
        lookback_df = df.iloc[current_idx-self.fvg_lookback:current_idx+1]
        
        # Look for a bearish FVG (3-candle formation)
        for i in range(2, self.fvg_lookback - 1):
            idx = -i - 1  # Convert to negative indexing from the end
            
            # Bearish FVG: High of the third candle is lower than the low of the first candle
            if lookback_df['High'].iloc[idx] < lookback_df['Low'].iloc[idx+2]:
                return idx
        
        return -1
    
    def calculate_position_size(self, risk_amount, price_risk):
        """Calculate position size based on risk percentage"""
        if price_risk <= 0:
            return 0
        
        # Calculate shares based on risk
        shares = risk_amount / price_risk
        
        # Round down to nearest whole share
        return int(shares)
    
    def process_long_entry(self, df, current_idx, fvg_idx):
        """Process a potential long entry"""
        lookback_df = df.iloc[current_idx-self.fvg_lookback:current_idx+1]
        
        # For long entry: top of the bullish FVG (high of the third candle)
        entry_price = lookback_df['High'].iloc[fvg_idx]
        
        # For long stop: low of the first candle in the FVG
        stop_loss = lookback_df['Low'].iloc[fvg_idx+2]
        
        # Calculate risk in price units
        risk = entry_price - stop_loss
        
        # Calculate take profit at specified R:R ratio
        take_profit = entry_price + (self.rr_ratio * risk)
        
        # Calculate risk amount in dollars
        risk_amount = self.current_balance * (self.risk_percent / 100)
        
        # Calculate position size
        position_size = self.calculate_position_size(risk_amount, risk)
        
        # Record the trade details
        trade = {
            'type': 'BUY LIMIT',
            'entry_time': df.index[current_idx],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'status': 'PENDING'
        }
        
        self.positions.append(trade)
        print(f"LONG order placed: Entry={entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}, Size={position_size}")
    
    def process_short_entry(self, df, current_idx, fvg_idx):
        """Process a potential short entry"""
        lookback_df = df.iloc[current_idx-self.fvg_lookback:current_idx+1]
        
        # For short entry: bottom of the bearish FVG (low of the third candle)
        entry_price = lookback_df['Low'].iloc[fvg_idx]
        
        # For short stop: high of the first candle in the FVG
        stop_loss = lookback_df['High'].iloc[fvg_idx+2]
        
        # Calculate risk in price units
        risk = stop_loss - entry_price
        
        # Calculate take profit at specified R:R ratio
        take_profit = entry_price - (self.rr_ratio * risk)
        
        # Calculate risk amount in dollars
        risk_amount = self.current_balance * (self.risk_percent / 100)
        
        # Calculate position size
        position_size = self.calculate_position_size(risk_amount, risk)
        
        # Record the trade details
        trade = {
            'type': 'SELL LIMIT',
            'entry_time': df.index[current_idx],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'status': 'PENDING'
        }
        
        self.positions.append(trade)
        print(f"SHORT order placed: Entry={entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}, Size={position_size}")
    
    def update_positions(self, df, current_idx):
        """Update status of pending and open positions"""
        current_bar = df.iloc[current_idx]
        
        # Copy positions to avoid modification during iteration
        positions_copy = self.positions.copy()
        self.positions = []
        
        for trade in positions_copy:
            # Skip already closed trades
            if trade['status'] == 'CLOSED':
                self.trade_history.append(trade)
                continue
            
            # Check if pending orders are filled
            if trade['status'] == 'PENDING':
                if trade['type'] == 'BUY LIMIT' and current_bar['Low'] <= trade['entry_price'] <= current_bar['High']:
                    trade['status'] = 'OPEN'
                    trade['fill_time'] = df.index[current_idx]
                    print(f"BUY LIMIT order filled at {trade['entry_price']:.2f}")
                    
                elif trade['type'] == 'SELL LIMIT' and current_bar['Low'] <= trade['entry_price'] <= current_bar['High']:
                    trade['status'] = 'OPEN'
                    trade['fill_time'] = df.index[current_idx]
                    print(f"SELL LIMIT order filled at {trade['entry_price']:.2f}")
            
            # Check if open orders hit stop loss or take profit
            if trade['status'] == 'OPEN':
                # For long positions
                if trade['type'] == 'BUY LIMIT':
                    # Check for stop loss hit
                    if current_bar['Low'] <= trade['stop_loss']:
                        trade['status'] = 'CLOSED'
                        trade['exit_time'] = df.index[current_idx]
                        trade['exit_price'] = trade['stop_loss']
                        trade['exit_reason'] = 'STOP_LOSS'
                        trade['profit'] = (trade['exit_price'] - trade['entry_price']) * trade['position_size']
                        self.current_balance += trade['profit']
                        print(f"LONG position stopped out at {trade['stop_loss']:.2f}, Profit: ${trade['profit']:.2f}")
                        self.trade_history.append(trade)
                        continue
                        
                    # Check for take profit hit
                    elif current_bar['High'] >= trade['take_profit']:
                        trade['status'] = 'CLOSED'
                        trade['exit_time'] = df.index[current_idx]
                        trade['exit_price'] = trade['take_profit']
                        trade['exit_reason'] = 'TAKE_PROFIT'
                        trade['profit'] = (trade['exit_price'] - trade['entry_price']) * trade['position_size']
                        self.current_balance += trade['profit']
                        print(f"LONG position take profit hit at {trade['take_profit']:.2f}, Profit: ${trade['profit']:.2f}")
                        self.trade_history.append(trade)
                        continue
                        
                # For short positions
                elif trade['type'] == 'SELL LIMIT':
                    # Check for stop loss hit
                    if current_bar['High'] >= trade['stop_loss']:
                        trade['status'] = 'CLOSED'
                        trade['exit_time'] = df.index[current_idx]
                        trade['exit_price'] = trade['stop_loss']
                        trade['exit_reason'] = 'STOP_LOSS'
                        trade['profit'] = (trade['entry_price'] - trade['exit_price']) * trade['position_size']
                        self.current_balance += trade['profit']
                        print(f"SHORT position stopped out at {trade['stop_loss']:.2f}, Profit: ${trade['profit']:.2f}")
                        self.trade_history.append(trade)
                        continue
                        
                    # Check for take profit hit
                    elif current_bar['Low'] <= trade['take_profit']:
                        trade['status'] = 'CLOSED'
                        trade['exit_time'] = df.index[current_idx]
                        trade['exit_price'] = trade['take_profit']
                        trade['exit_reason'] = 'TAKE_PROFIT'
                        trade['profit'] = (trade['entry_price'] - trade['exit_price']) * trade['position_size']
                        self.current_balance += trade['profit']
                        print(f"SHORT position take profit hit at {trade['take_profit']:.2f}, Profit: ${trade['profit']:.2f}")
                        self.trade_history.append(trade)
                        continue
            
            # If we reach here, the position is still active
            self.positions.append(trade)
    
    def backtest(self, data):
        """Run backtest on historical data"""
        equity_curve = [self.initial_balance]
        trade_days = []
        
        # Track if we have any active positions for the current day
        current_day = None
        has_position_today = False
        
        # Iterate through each bar
        for i in range(1, len(data)):
            # Get current timestamp
            timestamp = data.index[i]
            
            # Check if this is a new trading day
            if current_day != timestamp.date():
                current_day = timestamp.date()
                has_position_today = False
                trade_days.append(current_day)
            
            # Update existing positions
            self.update_positions(data, i)
            
            # Skip if we already have positions today or outside trading window
            if has_position_today or not self.is_trade_time(timestamp):
                equity_curve.append(self.current_balance)
                continue
            
            # Check for entry conditions
            has_liquidity_sweep = self.check_liquidity_sweep(data, i)
            has_strong_movement = self.check_strong_movement(data, i)
            
            # If we have at least one confluence
            if has_liquidity_sweep or has_strong_movement:
                # Look for FVGs
                bullish_fvg = self.find_bullish_fvg(data, i)
                bearish_fvg = self.find_bearish_fvg(data, i)
                
                # Process long entry if bullish FVG found
                if bullish_fvg >= 0:
                    self.process_long_entry(data, i, bullish_fvg)
                    has_position_today = True
                
                # Process short entry if bearish FVG found
                if bearish_fvg >= 0:
                    self.process_short_entry(data, i, bearish_fvg)
                    has_position_today = True
            
            equity_curve.append(self.current_balance)
        
        # Close out any remaining positions using the last bar
        remaining_positions = len(self.positions)
        if remaining_positions > 0:
            print(f"Closing {remaining_positions} positions at the end of the backtest period")
            
            last_bar = data.iloc[-1]
            for trade in self.positions:
                if trade['status'] == 'PENDING':
                    # Skip unfilled orders
                    trade['status'] = 'CANCELLED'
                elif trade['status'] == 'OPEN':
                    # Close at last price
                    trade['status'] = 'CLOSED'
                    trade['exit_time'] = data.index[-1]
                    trade['exit_price'] = last_bar['Close']
                    trade['exit_reason'] = 'END_OF_TEST'
                    
                    if trade['type'] == 'BUY LIMIT':
                        trade['profit'] = (trade['exit_price'] - trade['entry_price']) * trade['position_size']
                    else:
                        trade['profit'] = (trade['entry_price'] - trade['exit_price']) * trade['position_size']
                    
                    self.current_balance += trade['profit']
                
                self.trade_history.append(trade)
            
            self.positions = []
        
        # Calculate results
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
        losing_trades = sum(1 for trade in self.trade_history if trade.get('profit', 0) < 0)
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades * 100
            profit_factor = sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('profit', 0) > 0) / abs(sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('profit', 0) < 0) or 1)
            
            # Calculate average profit and loss
            avg_win = sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('profit', 0) > 0) / (winning_trades or 1)
            avg_loss = sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('profit', 0) < 0) / (losing_trades or 1)
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
        
        # Calculate drawdown
        max_balance = self.initial_balance
        max_drawdown = 0
        max_drawdown_pct = 0
        
        for i, balance in enumerate(equity_curve):
            max_balance = max(max_balance, balance)
            drawdown = max_balance - balance
            drawdown_pct = drawdown / max_balance * 100
            max_drawdown = max(max_drawdown, drawdown)
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        
        # Calculate final results
        net_profit = self.current_balance - self.initial_balance
        net_profit_pct = net_profit / self.initial_balance * 100
        
        # Create results dictionary
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'net_profit': net_profit,
            'net_profit_pct': net_profit_pct,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'equity_curve': equity_curve,
            'trade_history': self.trade_history,
            'trading_days': len(trade_days)
        }
        
        return results
    
    def plot_results(self, results):
        """Plot backtest results"""
        # Create figure with 2 subplots (equity curve and drawdown)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(results['equity_curve'])
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Account Balance ($)')
        ax1.grid(True)
        
        # Plot drawdown
        max_balance = self.initial_balance
        drawdown_curve = []
        
        for balance in results['equity_curve']:
            max_balance = max(max_balance, balance)
            drawdown_pct = (max_balance - balance) / max_balance * 100
            drawdown_curve.append(drawdown_pct)
        
        ax2.plot(drawdown_curve)
        ax2.set_title('Drawdown (%)')
        ax2.set_ylabel('Drawdown %')
        ax2.set_xlabel('Bars')
        ax2.invert_yaxis()  # Invert y-axis for better visualization
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Display performance metrics
        print("\n==== PERFORMANCE SUMMARY ====")
        print(f"Initial Balance: ${results['initial_balance']:.2f}")
        print(f"Final Balance: ${results['final_balance']:.2f}")
        print(f"Net Profit: ${results['net_profit']:.2f} ({results['net_profit_pct']:.2f}%)")
        print(f"Max Drawdown: ${results['max_drawdown']:.2f} ({results['max_drawdown_pct']:.2f}%)")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Avg. Win: ${results['avg_win']:.2f}")
        print(f"Avg. Loss: ${results['avg_loss']:.2f}")
        print(f"Trading Days: {results['trading_days']}")
        
        return fig

# Main execution
if __name__ == "__main__":
    # Load API key from environment variable or enter it manually
    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        api_key = input("Enter your Polygon.io API key: ")
    
    # Create strategy instance
    strategy = FVGStrategy(
        symbol='QQQ',
        risk_percent=1.0,
        rr_ratio=1.2,
        start_time='09:31',
        end_time='10:00',
        polygon_api_key=api_key
    )
    
    # Menu-driven approach to choose data source
    print("\nFVG Strategy Backtester (Polygon.io Edition)")
    print("-------------------------------------------")
    print("1. Fetch data from Polygon.io")
    print("2. Use custom CSV data")
    print("3. Use the provided sample data")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        # Get date range for backtest
        print("\nEnter date range for backtest (format: YYYY-MM-DD)")
        start_date = input("Start date: ")
        end_date = input("End date: ")
        
        # Get symbol to test
        symbol = input(f"Enter symbol to test (default: {strategy.symbol}): ")
        if symbol:
            strategy.symbol = symbol
        
        # Fetch data from Polygon.io
        data = strategy.fetch_polygon_data(start_date, end_date)
        if data is None or data.empty:
            print("No data retrieved from Polygon.io. Exiting.")
            exit()
    
    elif choice == '2':
        # Ask for CSV file path
        csv_path = input("Enter path to your CSV file: ")
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            exit()
        
        try:
            # Create a load_custom_data method similar to before
            def load_custom_data(self, filepath):
                print(f"Loading data from {filepath}...")
                df = pd.read_csv(filepath)
                
                # Convert Time/Datetime column to datetime
                time_col = None
                for col in ['Time', 'Datetime', 'Date', 'DateTime']:
                    if col in df.columns:
                        time_col = col
                        break
                
                if time_col is None:
                    raise ValueError("CSV must have a Time or Datetime column")
                
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.set_index(time_col)
                
                # Rename Last to Close if needed
                if 'Last' in df.columns and 'Close' not in df.columns:
                    df = df.rename(columns={'Last': 'Close'})
                
                # Ensure we have required columns
                required_columns = ['Open', 'High', 'Low', 'Close']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                print(f"Loaded data with {len(df)} bars")
                print(f"Date range: {df.index.min()} to {df.index.max()}")
                
                return df
            
            # Add method to class
            FVGStrategy.load_custom_data = load_custom_data
            
            # Load data
            data = strategy.load_custom_data(csv_path)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            exit()
    
    elif choice == '3':
        # Create a sample DataFrame from the provided data
        print("Using sample data from the screenshot...")
        # Sample data from the screenshot
        sample_data = [
            {'Time': '2025-01-08 10:09', 'Open': 21166.72, 'High': 21175.59, 'Low': 21164.66, 'Last': 21165.94},
            {'Time': '2025-01-08 10:10', 'Open': 21165.97, 'High': 21179.94, 'Low': 21121.87, 'Last': 21123.75},
            {'Time': '2025-01-08 10:11', 'Open': 21126.16, 'High': 21138.59, 'Low': 21116.91, 'Last': 21119.96},
            {'Time': '2025-01-08 10:12', 'Open': 21120.1, 'High': 21122.27, 'Low': 21103.3, 'Last': 21114.79},
            {'Time': '2025-01-08 10:13', 'Open': 21115.76, 'High': 21116.34, 'Low': 21078.98, 'Last': 21079.7},
            {'Time': '2025-01-08 10:14', 'Open': 21077.67, 'High': 21080.91, 'Low': 21059.2, 'Last': 21059.2},
            {'Time': '2025-01-08 10:15', 'Open': 21061.76, 'High': 21069.3, 'Low': 21052.33, 'Last': 21064.8},
            {'Time': '2025-01-08 10:16', 'Open': 21064.56, 'High': 21077.88, 'Low': 21056.6, 'Last': 21077.21},
            {'Time': '2025-01-08 10:17', 'Open': 21077.76, 'High': 21081.63, 'Low': 21070.4, 'Last': 21077.17},
            {'Time': '2025-01-08 10:18', 'Open': 21077.23, 'High': 21095.77, 'Low': 21077.23, 'Last': 21095.77},
            {'Time': '2025-01-08 10:19', 'Open': 21096.96, 'High': 21110.97, 'Low': 21095.9, 'Last': 21105.39},
            {'Time': '2025-01-08 10:20', 'Open': 21107.13, 'High': 21107.3, 'Low': 21074.03, 'Last': 21074.03},
            {'Time': '2025-01-08 10:21', 'Open': 21074.14, 'High': 21091.9, 'Low': 21074.14, 'Last': 21086.14},
            {'Time': '2025-01-08 10:22', 'Open': 21084.66, 'High': 21084.66, 'Low': 21059.15, 'Last': 21071.51},
            {'Time': '2025-01-08 10:23', 'Open': 21071.31, 'High': 21101.07, 'Low': 21071.31, 'Last': 21100.81},
            {'Time': '2025-01-08 10:24', 'Open': 21101.54, 'High': 21126.39, 'Low': 21099.17, 'Last': 21123.43},
            {'Time': '2025-01-08 10:25', 'Open': 21122.91, 'High': 21131.91, 'Low': 21122.91, 'Last': 21125.62}
        ]
        
        # Convert to DataFrame
        df = pd.DataFrame(sample_data)
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.set_index('Time')
        df = df.rename(columns={'Last': 'Close'})
        
        # Adjust strategy times to match sample data
        strategy.start_time = pd.to_datetime('10:09').time()
        strategy.end_time = pd.to_datetime('10:25').time()
        
        data = df
    
    else:
        print("Invalid choice. Exiting.")
        exit()
    
    # Run the backtest
    results = strategy.backtest(data)
    
    # Plot results
    fig = strategy.plot_results(results)
    plt.show()
    
    # Save trades to CSV if there are any
    if results['trade_history']:
        trades_df = pd.DataFrame(results['trade_history'])
        output_file = f"fvg_{strategy.symbol}_trade_history.csv"
        trades_df.to_csv(output_file, index=False)
        print(f"Trade history saved to {output_file}")