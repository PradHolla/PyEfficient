"""
================================================================================
                        DATA ENGINE MODULE (Person B)
================================================================================

PURPOSE:
    This module provides the mathematical foundation for the PyPairs 
    Statistical Arbitrage System. It handles:
    
    1. DATA MINING     - Download historical stock prices using yfinance
    2. CORRELATION     - Find the most correlated pair of stocks
    3. SPREAD CALC     - Calculate the hedge ratio using Linear Regression
    4. Z-SCORE         - Normalize the spread for trading signals

FLOW OVERVIEW:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Step 1: download_data(tickers)                                         │
    │     • Fetch Adjusted Close prices from Yahoo Finance                    │
    │     • Handle invalid tickers gracefully                                 │
    │     • Return a DataFrame with date index and ticker columns             │
    └───────────────────────────────┬─────────────────────────────────────────┘
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Step 2: find_correlated_pair(data)                                     │
    │     • Build a correlation matrix of all tickers                         │
    │     • Find the pair with highest correlation (excluding 1.0 diagonal)   │
    │     • Return tuple of (ticker_a, ticker_b, correlation_value)           │
    └───────────────────────────────┬─────────────────────────────────────────┘
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Step 3: calculate_zscore(pair_data)                                    │
    │     • Perform Linear Regression to get Hedge Ratio (slope)              │
    │     • Calculate Spread = Stock_A - (Hedge_Ratio * Stock_B)              │
    │     • Normalize: Z-Score = (Spread - Mean) / StdDev                     │
    │     • Return DataFrame with 'Spread' and 'Z-Score' columns              │
    └─────────────────────────────────────────────────────────────────────────┘

TRADING LOGIC (for context - implemented by other team members):
    • Z-Score > 2.0  → Spread is TOO WIDE  → Short Stock_A, Long Stock_B
    • Z-Score < -2.0 → Spread is TOO NARROW → Long Stock_A, Short Stock_B
    • Z-Score near 0 → Spread is NORMAL    → Close positions (Mean Reversion)

Author: Person B (Data & Math Engineer)
================================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, List, Optional


class DataEngine:
    """
    The DataEngine class provides all data fetching and mathematical
    calculations needed for the pairs trading strategy.
    
    Attributes:
        verbose (bool): If True, prints detailed debugging information
        
    Example Usage:
        engine = DataEngine(verbose=True)
        data = engine.download_data(['AAPL', 'MSFT', 'GOOGL', 'META'])
        pair = engine.find_correlated_pair(data)
        result = engine.calculate_zscore(data[[pair[0], pair[1]]])
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the DataEngine.
        
        Args:
            verbose: If True, prints debugging statements throughout execution
        """
        self.verbose = verbose
        self._debug_print("=" * 70)
        self._debug_print("DataEngine initialized")
        self._debug_print(f"Verbose mode: {self.verbose}")
        self._debug_print("=" * 70)
    
    def _debug_print(self, message: str) -> None:
        """
        Helper function to print debug messages only when verbose mode is on.
        
        Args:
            message: The debug message to print
        """
        if self.verbose:
            print(f"[DEBUG] {message}")
    
    # ==========================================================================
    #                          STEP 1: DOWNLOAD DATA
    # ==========================================================================
    def download_data(
        self, 
        tickers: List[str], 
        period: str = "2y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Download historical Adjusted Close prices from Yahoo Finance.
        
        FLOW:
        1. Validate input tickers list
        2. Download data using yfinance
        3. Extract 'Adj Close' column (accounts for splits/dividends)
        4. Handle any invalid tickers by removing them
        5. Drop rows with NaN values
        6. Return clean DataFrame
        
        Args:
            tickers: List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
            period: Time period to download ('1y', '2y', '5y', etc.)
            interval: Data interval ('1d' for daily, '1h' for hourly, etc.)
            
        Returns:
            pd.DataFrame: DataFrame with Date index and Adj Close prices
            
        Raises:
            ValueError: If less than 2 valid tickers are provided
        """
        self._debug_print("\n" + "=" * 70)
        self._debug_print("STEP 1: DOWNLOADING STOCK DATA")
        self._debug_print("=" * 70)
        
        # -------------------- Input Validation --------------------
        self._debug_print(f"Received tickers: {tickers}")
        self._debug_print(f"Period: {period}, Interval: {interval}")
        
        if len(tickers) < 2:
            raise ValueError("Need at least 2 tickers for pairs trading!")
        
        # -------------------- Download from yfinance --------------------
        self._debug_print("\nConnecting to Yahoo Finance...")
        self._debug_print("Downloading Adjusted Close prices...")
        
        try:
            # yf.download returns a DataFrame with MultiIndex columns if 
            # multiple tickers are provided: (Price Type, Ticker)
            raw_data = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                progress=False,  # Disable progress bar for cleaner output
                auto_adjust=False  # Keep 'Adj Close' separate from 'Close'
            )
            
            self._debug_print(f"Raw data shape: {raw_data.shape}")
            self._debug_print(f"Raw data columns: {list(raw_data.columns)}")
            
        except Exception as e:
            self._debug_print(f"ERROR downloading data: {e}")
            raise
        
        # -------------------- Extract Adjusted Close --------------------
        self._debug_print("\nExtracting 'Adj Close' prices...")
        
        # Handle both single ticker (returns Series) and multiple tickers
        if len(tickers) == 1:
            # Single ticker returns simple columns
            if 'Adj Close' in raw_data.columns:
                adj_close = raw_data[['Adj Close']]
                adj_close.columns = tickers
            else:
                raise ValueError(f"No 'Adj Close' data for {tickers[0]}")
        else:
            # Multiple tickers return MultiIndex columns
            if 'Adj Close' in raw_data.columns.get_level_values(0):
                adj_close = raw_data['Adj Close']
            else:
                # Try 'Close' as fallback (newer yfinance versions)
                self._debug_print("'Adj Close' not found, trying 'Close'...")
                adj_close = raw_data['Close']
        
        self._debug_print(f"Adjusted Close data shape: {adj_close.shape}")
        
        # -------------------- Handle Invalid Tickers --------------------
        self._debug_print("\nChecking for invalid tickers...")
        
        # Find columns that are entirely NaN (invalid tickers)
        valid_tickers = []
        invalid_tickers = []
        
        for ticker in adj_close.columns:
            if adj_close[ticker].isna().all():
                invalid_tickers.append(ticker)
                self._debug_print(f"  ✗ {ticker}: INVALID (no data)")
            else:
                valid_tickers.append(ticker)
                self._debug_print(f"  ✓ {ticker}: Valid ({adj_close[ticker].notna().sum()} data points)")
        
        if invalid_tickers:
            self._debug_print(f"\nWARNING: Removing invalid tickers: {invalid_tickers}")
            adj_close = adj_close[valid_tickers]
        
        # Check if we still have enough tickers
        if len(valid_tickers) < 2:
            raise ValueError(
                f"Only {len(valid_tickers)} valid ticker(s) found. "
                f"Need at least 2 for pairs trading!"
            )
        
        # -------------------- Clean Data --------------------
        self._debug_print("\nCleaning data (dropping NaN rows)...")
        
        rows_before = len(adj_close)
        adj_close = adj_close.dropna()
        rows_after = len(adj_close)
        
        self._debug_print(f"Rows before cleaning: {rows_before}")
        self._debug_print(f"Rows after cleaning: {rows_after}")
        self._debug_print(f"Rows dropped: {rows_before - rows_after}")
        
        # -------------------- Summary --------------------
        self._debug_print("\n" + "-" * 50)
        self._debug_print("DOWNLOAD COMPLETE - SUMMARY")
        self._debug_print("-" * 50)
        self._debug_print(f"Valid Tickers: {list(adj_close.columns)}")
        self._debug_print(f"Date Range: {adj_close.index[0]} to {adj_close.index[-1]}")
        self._debug_print(f"Total Trading Days: {len(adj_close)}")
        self._debug_print(f"DataFrame Shape: {adj_close.shape}")
        self._debug_print("-" * 50)
        
        return adj_close
    
    # ==========================================================================
    #                      STEP 2: FIND CORRELATED PAIR
    # ==========================================================================
    def find_correlated_pair(
        self, 
        data: pd.DataFrame
    ) -> Tuple[str, str, float]:
        """
        Find the two stocks with the highest correlation.
        
        FLOW:
        1. Calculate the correlation matrix using pandas
        2. Mask the diagonal (self-correlation = 1.0) and upper triangle
        3. Find the maximum correlation value
        4. Get the row/column names (tickers) for that maximum
        5. Return the pair and their correlation
        
        WHY THIS MATTERS:
        Pairs trading works best when two stocks move together. If Ford and GM
        have 0.95 correlation, when one goes up, the other usually does too.
        If they temporarily diverge, we bet they'll come back together!
        
        Args:
            data: DataFrame with stock prices (columns = tickers)
            
        Returns:
            Tuple of (ticker_a, ticker_b, correlation_value)
            
        Raises:
            ValueError: If less than 2 columns in data
        """
        self._debug_print("\n" + "=" * 70)
        self._debug_print("STEP 2: FINDING MOST CORRELATED PAIR")
        self._debug_print("=" * 70)
        
        # -------------------- Input Validation --------------------
        if data.shape[1] < 2:
            raise ValueError("Need at least 2 stocks to find a pair!")
        
        self._debug_print(f"Analyzing {data.shape[1]} stocks: {list(data.columns)}")
        
        # -------------------- Calculate Correlation Matrix --------------------
        self._debug_print("\nCalculating Pearson correlation matrix...")
        
        # Pearson correlation: measures linear relationship between -1 and +1
        # +1 = perfect positive correlation (move together)
        #  0 = no correlation (independent)
        # -1 = perfect negative correlation (move opposite)
        corr_matrix = data.corr(method='pearson')
        
        self._debug_print("\nCorrelation Matrix:")
        self._debug_print("-" * 50)
        # Print correlation matrix in a nice format
        for ticker in corr_matrix.columns:
            row_values = [f"{corr_matrix.loc[ticker, col]:.4f}" 
                         for col in corr_matrix.columns]
            self._debug_print(f"  {ticker:6s}: {row_values}")
        self._debug_print("-" * 50)
        
        # -------------------- Find Maximum Correlation --------------------
        self._debug_print("\nFinding highest correlation (excluding self-correlation)...")
        
        # Create a mask to ignore:
        # 1. Diagonal elements (self-correlation = 1.0)
        # 2. Upper triangle (to avoid counting pairs twice)
        # 
        # Example for 3 stocks (A, B, C):
        #        A     B     C
        #   A  [1.0]  0.8   0.7    <- upper triangle + diagonal
        #   B   0.8  [1.0]  0.9    <- we only want lower triangle
        #   C   0.7   0.9  [1.0]
        #
        # We want: (B,A)=0.8, (C,A)=0.7, (C,B)=0.9
        
        # np.tril creates lower triangle, k=-1 excludes diagonal
        mask = np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool)
        
        # Apply mask - set upper triangle and diagonal to NaN
        masked_corr = corr_matrix.where(mask)
        
        self._debug_print("Masked correlation matrix (lower triangle only):")
        self._debug_print(masked_corr.to_string())
        
        # Find the maximum correlation value
        max_corr = masked_corr.max().max()
        self._debug_print(f"\nMaximum correlation found: {max_corr:.6f}")
        
        # Find which pair has this maximum correlation
        # stack() converts matrix to Series with MultiIndex (row, col)
        max_pair = masked_corr.stack().idxmax()
        
        ticker_a = max_pair[0]  # Row index
        ticker_b = max_pair[1]  # Column index
        
        # -------------------- Summary --------------------
        self._debug_print("\n" + "-" * 50)
        self._debug_print("PAIR FINDING COMPLETE - SUMMARY")
        self._debug_print("-" * 50)
        self._debug_print(f"Most Correlated Pair: {ticker_a} & {ticker_b}")
        self._debug_print(f"Correlation: {max_corr:.6f}")
        self._debug_print(f"Interpretation: These stocks move together {max_corr*100:.1f}% of the time")
        self._debug_print("-" * 50)
        
        return (ticker_a, ticker_b, max_corr)
    
    # ==========================================================================
    #                      STEP 3: CALCULATE Z-SCORE
    # ==========================================================================
    def calculate_zscore(
        self, 
        pair_data: pd.DataFrame,
        lookback_window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate the Z-Score of the spread between two stocks.
        
        FLOW:
        1. Perform Linear Regression to find Hedge Ratio (slope)
           - Stock_A = slope * Stock_B + intercept
           - The slope tells us how many shares of B to trade per share of A
           
        2. Calculate Spread
           - Spread = Stock_A - (Hedge_Ratio * Stock_B)
           - This is the "distance" between the stocks
           
        3. Calculate Z-Score
           - Z-Score = (Spread - Mean) / Standard_Deviation
           - This normalizes the spread to a standard scale
           - Z-Score of 0 = spread is at its average
           - Z-Score of 2 = spread is 2 standard deviations above average
        
        WHY Z-SCORE?
        Raw spread values vary wildly between different stock pairs.
        $AAPL spread might be $50, while $F spread might be $2.
        Z-Score normalizes everything to the same scale for easy comparison.
        
        Args:
            pair_data: DataFrame with exactly 2 columns (the pair of stocks)
            lookback_window: Rolling window for mean/std calculation (None = all data)
            
        Returns:
            pd.DataFrame: Original data plus 'Hedge_Ratio', 'Spread', and 'Z-Score' columns
            
        Raises:
            ValueError: If pair_data doesn't have exactly 2 columns
        """
        self._debug_print("\n" + "=" * 70)
        self._debug_print("STEP 3: CALCULATING Z-SCORE")
        self._debug_print("=" * 70)
        
        # -------------------- Input Validation --------------------
        if pair_data.shape[1] != 2:
            raise ValueError(
                f"Expected exactly 2 columns (a pair), got {pair_data.shape[1]}"
            )
        
        # Get the ticker names
        ticker_a = pair_data.columns[0]
        ticker_b = pair_data.columns[1]
        
        self._debug_print(f"Stock A (Dependent Variable): {ticker_a}")
        self._debug_print(f"Stock B (Independent Variable): {ticker_b}")
        self._debug_print(f"Data points: {len(pair_data)}")
        
        # -------------------- Linear Regression for Hedge Ratio --------------------
        self._debug_print("\n" + "-" * 50)
        self._debug_print("SUBSTEP 3.1: LINEAR REGRESSION")
        self._debug_print("-" * 50)
        
        # Extract price series
        price_a = pair_data[ticker_a].values
        price_b = pair_data[ticker_b].values
        
        self._debug_print(f"{ticker_a} price range: ${price_a.min():.2f} - ${price_a.max():.2f}")
        self._debug_print(f"{ticker_b} price range: ${price_b.min():.2f} - ${price_b.max():.2f}")
        
        # Perform Linear Regression using numpy.polyfit
        # polyfit(x, y, degree) returns coefficients [slope, intercept] for degree=1
        #
        # We're fitting: Stock_A = slope * Stock_B + intercept
        #
        # The SLOPE is our HEDGE RATIO - it tells us how many shares of Stock_B
        # we need to trade to hedge one share of Stock_A
        
        self._debug_print("\nUsing numpy.polyfit for linear regression...")
        self._debug_print(f"Model: {ticker_a} = slope × {ticker_b} + intercept")
        
        # polyfit returns [slope, intercept] for degree 1 polynomial
        coefficients = np.polyfit(price_b, price_a, deg=1)
        
        hedge_ratio = coefficients[0]  # Slope
        intercept = coefficients[1]    # Y-intercept
        
        self._debug_print(f"\nRegression Results:")
        self._debug_print(f"  Hedge Ratio (slope): {hedge_ratio:.6f}")
        self._debug_print(f"  Intercept: {intercept:.6f}")
        self._debug_print(f"\nInterpretation:")
        self._debug_print(f"  For every 1 share of {ticker_a}, trade {hedge_ratio:.4f} shares of {ticker_b}")
        
        # -------------------- Calculate Spread --------------------
        self._debug_print("\n" + "-" * 50)
        self._debug_print("SUBSTEP 3.2: CALCULATE SPREAD")
        self._debug_print("-" * 50)
        
        # Spread = Stock_A - (Hedge_Ratio × Stock_B)
        #
        # If stocks are perfectly correlated, spread should be nearly constant
        # When spread widens → stocks have diverged → opportunity!
        
        spread = price_a - (hedge_ratio * price_b)
        
        self._debug_print(f"Formula: Spread = {ticker_a} - ({hedge_ratio:.4f} × {ticker_b})")
        self._debug_print(f"\nSpread Statistics:")
        self._debug_print(f"  Mean: {spread.mean():.4f}")
        self._debug_print(f"  Std Dev: {spread.std():.4f}")
        self._debug_print(f"  Min: {spread.min():.4f}")
        self._debug_print(f"  Max: {spread.max():.4f}")
        
        # -------------------- Calculate Z-Score --------------------
        self._debug_print("\n" + "-" * 50)
        self._debug_print("SUBSTEP 3.3: CALCULATE Z-SCORE")
        self._debug_print("-" * 50)
        
        # Z-Score Formula: Z = (X - μ) / σ
        # Where:
        #   X = current spread value
        #   μ = mean of spread
        #   σ = standard deviation of spread
        
        if lookback_window is not None:
            # Rolling Z-Score (more responsive to recent data)
            self._debug_print(f"Using ROLLING window of {lookback_window} days")
            spread_series = pd.Series(spread, index=pair_data.index)
            spread_mean = spread_series.rolling(window=lookback_window).mean()
            spread_std = spread_series.rolling(window=lookback_window).std()
            zscore = (spread_series - spread_mean) / spread_std
        else:
            # Static Z-Score (using all historical data)
            self._debug_print("Using STATIC mean/std (all historical data)")
            spread_mean = spread.mean()
            spread_std = spread.std()
            zscore = (spread - spread_mean) / spread_std
        
        self._debug_print(f"\nZ-Score Formula: Z = (Spread - {spread_mean if isinstance(spread_mean, float) else 'rolling_mean'}) / {spread_std if isinstance(spread_std, float) else 'rolling_std'}")
        self._debug_print(f"\nZ-Score Statistics:")
        zscore_array = zscore.values if hasattr(zscore, 'values') else zscore
        self._debug_print(f"  Mean: {np.nanmean(zscore_array):.4f} (should be ~0)")
        self._debug_print(f"  Std Dev: {np.nanstd(zscore_array):.4f} (should be ~1)")
        self._debug_print(f"  Min: {np.nanmin(zscore_array):.4f}")
        self._debug_print(f"  Max: {np.nanmax(zscore_array):.4f}")
        
        # -------------------- Build Result DataFrame --------------------
        self._debug_print("\n" + "-" * 50)
        self._debug_print("BUILDING RESULT DATAFRAME")
        self._debug_print("-" * 50)
        
        # Create a copy of the input data
        result = pair_data.copy()
        
        # Add calculated columns
        result['Hedge_Ratio'] = hedge_ratio
        result['Spread'] = spread
        result['Z-Score'] = zscore
        
        self._debug_print(f"Columns in result: {list(result.columns)}")
        self._debug_print(f"Result shape: {result.shape}")
        
        # -------------------- Trading Signal Analysis --------------------
        self._debug_print("\n" + "-" * 50)
        self._debug_print("TRADING SIGNAL ANALYSIS")
        self._debug_print("-" * 50)
        
        zscore_values = result['Z-Score'].dropna()
        
        # Count potential trading signals
        long_signals = (zscore_values < -2.0).sum()
        short_signals = (zscore_values > 2.0).sum()
        neutral = ((zscore_values >= -2.0) & (zscore_values <= 2.0)).sum()
        
        self._debug_print(f"Trading Signals (|Z-Score| > 2.0 threshold):")
        self._debug_print(f"  📈 Long {ticker_a} / Short {ticker_b} (Z < -2.0): {long_signals} days")
        self._debug_print(f"  📉 Short {ticker_a} / Long {ticker_b} (Z > 2.0): {short_signals} days")
        self._debug_print(f"  ⏸️  No trade (|Z| <= 2.0): {neutral} days")
        
        current_zscore = zscore_values.iloc[-1] if len(zscore_values) > 0 else np.nan
        self._debug_print(f"\nCurrent Z-Score (latest): {current_zscore:.4f}")
        
        if current_zscore > 2.0:
            self._debug_print(f"🚨 SIGNAL: Consider SHORT {ticker_a}, LONG {ticker_b}")
        elif current_zscore < -2.0:
            self._debug_print(f"🚨 SIGNAL: Consider LONG {ticker_a}, SHORT {ticker_b}")
        else:
            self._debug_print(f"✅ No trade signal - spread is within normal range")
        
        # -------------------- Summary --------------------
        self._debug_print("\n" + "=" * 70)
        self._debug_print("Z-SCORE CALCULATION COMPLETE - FINAL SUMMARY")
        self._debug_print("=" * 70)
        self._debug_print(f"Pair: {ticker_a} & {ticker_b}")
        self._debug_print(f"Hedge Ratio: {hedge_ratio:.6f}")
        self._debug_print(f"Output DataFrame contains: {list(result.columns)}")
        self._debug_print(f"Ready for trading strategy execution!")
        self._debug_print("=" * 70)
        
        return result
    
    # ==========================================================================
    #                      CONVENIENCE METHOD: FULL PIPELINE
    # ==========================================================================
    def run_full_pipeline(
        self, 
        tickers: List[str],
        period: str = "2y"
    ) -> Tuple[pd.DataFrame, Tuple[str, str, float]]:
        """
        Convenience method to run the complete pipeline.
        
        This runs all three steps in sequence:
        1. Download data
        2. Find correlated pair
        3. Calculate Z-Score
        
        Args:
            tickers: List of stock tickers to analyze
            period: Historical data period
            
        Returns:
            Tuple of (result_dataframe, pair_info)
            - result_dataframe: DataFrame with Z-Score column
            - pair_info: Tuple of (ticker_a, ticker_b, correlation)
        """
        self._debug_print("\n" + "#" * 70)
        self._debug_print("#" + " " * 20 + "FULL PIPELINE EXECUTION" + " " * 23 + "#")
        self._debug_print("#" * 70)
        
        # Step 1: Download
        data = self.download_data(tickers, period=period)
        
        # Step 2: Find pair
        pair_info = self.find_correlated_pair(data)
        ticker_a, ticker_b, correlation = pair_info
        
        # Step 3: Calculate Z-Score
        pair_data = data[[ticker_a, ticker_b]]
        result = self.calculate_zscore(pair_data)
        
        self._debug_print("\n" + "#" * 70)
        self._debug_print("#" + " " * 22 + "PIPELINE COMPLETE" + " " * 27 + "#")
        self._debug_print("#" * 70)
        
        return result, pair_info


# ==============================================================================
#                              MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    """
    Demo execution showing how to use the DataEngine class.
    
    This demonstrates the complete flow from data download to Z-Score calculation.
    """
    
    print("\n" + "🚀" * 35)
    print("\n   PyPairs Statistical Arbitrage System - Data Engine Demo\n")
    print("🚀" * 35 + "\n")
    
    # Initialize the engine with verbose output
    engine = DataEngine(verbose=True)
    
    # Define a list of automotive and tech stocks for demonstration
    # Automotive: Ford (F), General Motors (GM)
    # Tech: Apple (AAPL), Microsoft (MSFT)
    demo_tickers = ['F', 'GM', 'AAPL', 'MSFT']
    
    print(f"\nDemo will analyze: {demo_tickers}")
    print("Looking for the most correlated pair...\n")
    
    try:
        # Run the full pipeline
        result, pair_info = engine.run_full_pipeline(demo_tickers, period="1y")
        
        # Display final results
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"\nBest Pair Found: {pair_info[0]} & {pair_info[1]}")
        print(f"Correlation: {pair_info[2]:.4f}")
        print(f"\nLast 5 rows of output:")
        print(result.tail())
        
        print("\n" + "=" * 70)
        print("SUCCESS! The Z-Score DataFrame is ready for trading strategy!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        raise
