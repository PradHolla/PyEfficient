"""
Backtester for PyPairs (Person C)

Implements a simple pairs-trading backtester that consumes the DataEngine
output (a DataFrame with two price columns and a 'Z-Score' column) and
simulates entry/exit rules based on Z-Score thresholds.

Author: Person C (Strategy Developer)
"""
from typing import Optional

import numpy as np
import pandas as pd


class Backtester:
    """Simple pairs trading backtester.

    Methods
    -------
    run_backtest(data, entry_threshold=2.0, exit_threshold=0.0)
        Run the backtest on a DataFrame produced by DataEngine.calculate_zscore
    """

    def __init__(self):
        pass

    def run_backtest(
        self,
        data: pd.DataFrame,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        position_size: float = 1.0,
    ) -> pd.DataFrame:
        """
        Run a simple backtest using Z-Score entry/exit rules.

        Assumptions and contract:
        - `data` contains two price columns (the original tickers as the
          first two columns) and a 'Z-Score' column. It may also contain
          'Hedge_Ratio'. If 'Hedge_Ratio' is a scalar or series, it will be
          used to size the hedge leg.
        - Positions are sized as: +1 unit of A and -hedge_ratio units of B
          for a "long" A / "short" B position. The opposite applies for
          a short A / long B position.
        - PnL is calculated using close-to-close price changes. The position
          decided at time t is applied to the PnL from t -> t+1. To avoid
          lookahead, the implementation shifts positions by one day when
          computing daily PnL.

        Args:
            data: DataFrame with at least two price columns and 'Z-Score'.
            entry_threshold: Z-Score magnitude to open a trade (default 2.0).
            exit_threshold: Z-Score magnitude to close a trade (default 0.0).
            position_size: Multiplier for position size (default 1.0).

        Returns:
            pd.DataFrame: Copy of `data` with added columns:
                - 'Position' : current position state (-1, 0, 1)
                - 'Daily PnL' : PnL for each day
                - 'Cumulative PnL' : running total of PnL
        """

        # Basic validation
        if 'Z-Score' not in data.columns:
            raise ValueError("Input DataFrame must contain a 'Z-Score' column")

        if data.shape[1] < 3:
            # expect at least two price columns + other metrics
            raise ValueError(
                "Input DataFrame must contain at least two price columns"
            )

        # Work on a copy
        df = data.copy()

        # Assume first two columns are price series for stock A and stock B
        price_col_a = df.columns[0]
        price_col_b = df.columns[1]

        # Hedge ratio: prefer a column in DF, otherwise try to treat as scalar
        if 'Hedge_Ratio' in df.columns:
            hedge_ratio = df['Hedge_Ratio']
            # If hedge_ratio is a scalar series (same value for all rows), keep as series
        else:
            # default to 1.0 if not provided
            hedge_ratio = 1.0

        # Compute price deltas (close-to-close)
        df['_delta_a'] = df[price_col_a].diff()
        df['_delta_b'] = df[price_col_b].diff()

        # Determine position at each row from Z-Score
        z = df['Z-Score']

        # Initialize positions to 0 (flat) and then set from row 1 onwards
        position = pd.Series(0, index=df.index, dtype=int)

        # Build positions with simple rules (no transaction costs, no slippage)
        for i in range(1, len(df)):
            zs = z.iloc[i]
            prev_pos = position.iloc[i - 1]

            # If signal is NaN, carry previous position
            if zs is None or np.isnan(zs):
                position.iloc[i] = prev_pos
                continue

            # Entry rules
            if zs > entry_threshold:
                # Spread is high: SHORT A, LONG B -> state = -1
                position.iloc[i] = -1
            elif zs < -entry_threshold:
                # Spread is low: LONG A, SHORT B -> state = 1
                position.iloc[i] = 1
            else:
                # Exit rule: if Z-Score magnitude crosses below exit_threshold, close
                if abs(zs) < exit_threshold:
                    position.iloc[i] = 0
                else:
                    # Otherwise hold previous position
                    position.iloc[i] = prev_pos

        df['Position'] = position

        # Shift position to avoid look-ahead: today's PnL uses yesterday's position
        pos_for_pnl = df['Position'].shift(1).fillna(0)

        # If hedge_ratio is a Series align it, otherwise broadcast scalar
        if isinstance(hedge_ratio, pd.Series):
            hr = hedge_ratio
        else:
            hr = hedge_ratio

        # Compute daily PnL: position * (delta_A - hedge_ratio * delta_B) * position_size
        if isinstance(hr, pd.Series):
            pnl = pos_for_pnl * (df['_delta_a'] - hr * df['_delta_b']) * position_size
        else:
            pnl = pos_for_pnl * (df['_delta_a'] - hr * df['_delta_b']) * position_size

        df['Daily PnL'] = pnl.fillna(0.0)
        df['Cumulative PnL'] = df['Daily PnL'].cumsum()

        # Clean helper cols
        df = df.drop(columns=['_delta_a', '_delta_b'])

        return df


if __name__ == '__main__':
    # Mandatory behavior: always run DataEngine to produce Z-Score output and then backtest.
    import sys
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Ensure project root on sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from pypairs.data_engine import DataEngine
    except Exception as exc:
        print('Failed to import DataEngine. Ensure Person B (data_engine.py) is present.')
        raise

    print('Running DataEngine (mandatory) to compute Z-Score and pair...')
    engine = DataEngine(verbose=False)
    # Default tickers — adjust as needed
    demo_tickers = ['F', 'GM', 'AAPL', 'MSFT']
    result_df, pair_info = engine.run_full_pipeline(demo_tickers, period='1y')

    ticker_a, ticker_b, corr = pair_info
    print(f'Pair found: {ticker_a} & {ticker_b} (corr={corr:.4f})')

    # Run backtest on the freshly produced DataEngine output
    bt = Backtester()
    backtest_df = bt.run_backtest(result_df, entry_threshold=2.0, exit_threshold=0.0)

    print('\nRows in dataset:', len(backtest_df))
    cols_to_show = [backtest_df.columns[0], backtest_df.columns[1], 'Z-Score', 'Position', 'Cumulative PnL']
    cols_to_show = [c for c in cols_to_show if c in backtest_df.columns]
    print('\nLast 5 rows:')
    print(backtest_df[cols_to_show].tail().to_string())
    print('\nFinal Cumulative PnL: ${:.4f}'.format(backtest_df['Cumulative PnL'].iloc[-1]))
