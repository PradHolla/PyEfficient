"""
Visualizer for PyPairs (Person A)

Generates a 3-panel dashboard displaying Z-Score signals, equity curve,
and optimization results for the pairs trading strategy.

Author: Person A (The Architect - Integration & Visuals)
"""
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:
    """Dashboard visualization for pairs trading strategy.

    Methods
    -------
    plot_dashboard(backtest_df, optimization_df, pair_names, save_path=None)
        Generate a 3-panel dashboard with Z-Score, PnL, and optimization results.
    """

    def __init__(self):
        """Initialize the Visualizer with default styling."""
        # Set a clean style for the plots
        plt.style.use('seaborn-v0_8-whitegrid')

    def plot_dashboard(
        self,
        backtest_df: pd.DataFrame,
        optimization_df: pd.DataFrame,
        pair_names: Tuple[str, str],
        entry_threshold: float = 2.0,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Generate a 3-panel dashboard for the pairs trading strategy.

        Parameters
        ----------
        backtest_df : pd.DataFrame
            DataFrame from Backtester.run_backtest() containing:
            - 'Z-Score': The normalized spread signal
            - 'Cumulative PnL': Running total of profit/loss
            - 'Position': Current position state (-1, 0, 1)
        optimization_df : pd.DataFrame
            DataFrame from StrategyOptimizer.optimize_thresholds() containing:
            - 'Threshold': Entry threshold values tested
            - 'Final Profit': Final cumulative PnL for each threshold
        pair_names : Tuple[str, str]
            Names of the two stocks in the pair (e.g., ('F', 'GM'))
        entry_threshold : float, optional
            The entry threshold used, for display purposes (default 2.0)
        save_path : str, optional
            If provided, save the figure to this path instead of displaying

        Returns
        -------
        None
            Displays the plot or saves to file
        """
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(
            f'PyPairs Dashboard: {pair_names[0]} & {pair_names[1]}',
            fontsize=14,
            fontweight='bold'
        )

        # =====================================================================
        # Panel 1 (Top): Z-Score Time Series with Entry Zones
        # =====================================================================
        ax1 = axes[0]
        ax1.plot(
            backtest_df.index,
            backtest_df['Z-Score'],
            label='Z-Score',
            color='blue',
            linewidth=1
        )

        # Draw entry threshold lines
        ax1.axhline(
            y=entry_threshold,
            color='red',
            linestyle='--',
            linewidth=1.5,
            label=f'Short Entry (+{entry_threshold})'
        )
        ax1.axhline(
            y=-entry_threshold,
            color='green',
            linestyle='--',
            linewidth=1.5,
            label=f'Long Entry (-{entry_threshold})'
        )
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)

        # Fill entry zones for visual clarity
        ax1.fill_between(
            backtest_df.index,
            entry_threshold,
            backtest_df['Z-Score'].max() + 0.5,
            alpha=0.1,
            color='red',
            label='Short Zone'
        )
        ax1.fill_between(
            backtest_df.index,
            -entry_threshold,
            backtest_df['Z-Score'].min() - 0.5,
            alpha=0.1,
            color='green',
            label='Long Zone'
        )

        ax1.set_title('Z-Score Signal with Entry Zones', fontsize=11)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Z-Score')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # =====================================================================
        # Panel 2 (Middle): Cumulative PnL (Equity Curve)
        # =====================================================================
        ax2 = axes[1]
        ax2.plot(
            backtest_df.index,
            backtest_df['Cumulative PnL'],
            label='Cumulative PnL',
            color='darkgreen',
            linewidth=1.5
        )
        ax2.fill_between(
            backtest_df.index,
            0,
            backtest_df['Cumulative PnL'],
            alpha=0.3,
            color='green',
            where=(backtest_df['Cumulative PnL'] >= 0)
        )
        ax2.fill_between(
            backtest_df.index,
            0,
            backtest_df['Cumulative PnL'],
            alpha=0.3,
            color='red',
            where=(backtest_df['Cumulative PnL'] < 0)
        )
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        final_pnl = backtest_df['Cumulative PnL'].iloc[-1]
        ax2.set_title(
            f'Equity Curve (Final PnL: ${final_pnl:.2f})',
            fontsize=11
        )
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative PnL ($)')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # =====================================================================
        # Panel 3 (Bottom): Optimization Results Bar Chart
        # =====================================================================
        ax3 = axes[2]
        thresholds = optimization_df['Threshold']
        profits = optimization_df['Final Profit']

        # Color bars based on profit (green = positive, red = negative)
        colors = ['green' if p >= 0 else 'red' for p in profits]

        bars = ax3.bar(
            thresholds.astype(str),
            profits,
            color=colors,
            edgecolor='black',
            alpha=0.7
        )

        # Highlight the best threshold
        best_idx = profits.idxmax()
        best_threshold = thresholds.loc[best_idx]
        best_profit = profits.loc[best_idx]

        # Find the bar index for highlighting
        bar_idx = list(thresholds).index(best_threshold)
        bars[bar_idx].set_edgecolor('gold')
        bars[bar_idx].set_linewidth(3)

        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title(
            f'Optimization Results (Best: Threshold={best_threshold}, Profit=${best_profit:.2f})',
            fontsize=11
        )
        ax3.set_xlabel('Entry Threshold')
        ax3.set_ylabel('Final Profit ($)')
        ax3.grid(True, alpha=0.3, axis='y')

        # Rotate x-axis labels for readability
        ax3.tick_params(axis='x', rotation=45)

        # =====================================================================
        # Final adjustments and display
        # =====================================================================
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Dashboard saved to: {save_path}')
        else:
            plt.show()

        plt.close(fig)
