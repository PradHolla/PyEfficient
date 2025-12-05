"""
Main Entry Point for PyPairs (Person A)

Orchestrates the full pairs trading pipeline:
1. Data Ingestion & Pair Selection (DataEngine)
2. Strategy Optimization (StrategyOptimizer)
3. Final Backtest with Optimal Threshold (Backtester)
4. Risk Analysis (RiskMetrics)
5. Dashboard Visualization (Visualizer)

Author: Person A (The Architect - Integration & Visuals)
"""
import sys

from pypairs.data_engine import DataEngine
from pypairs.backtester import Backtester
from pypairs.risk_engine import RiskMetrics
from pypairs.optimizer import StrategyOptimizer
from pypairs.visualizer import Visualizer


def print_header(stage_num: int, title: str) -> None:
    """Print a formatted stage header."""
    print(f"\n{'='*60}")
    print(f"  Stage {stage_num}: {title}")
    print(f"{'='*60}")


def print_summary(pair_info, best_threshold, best_profit, final_pnl, metrics) -> None:
    """Print the final summary report."""
    ticker_a, ticker_b, correlation = pair_info

    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*18 + "PYPAIRS SUMMARY REPORT" + " "*18 + "║")
    print("╠" + "═"*58 + "╣")
    print(f"║  Pair Selected: {ticker_a} & {ticker_b}".ljust(59) + "║")
    print(f"║  Correlation: {correlation:.4f}".ljust(59) + "║")
    print("╠" + "═"*58 + "╣")
    print(f"║  Best Entry Threshold: {best_threshold}".ljust(59) + "║")
    print(f"║  Optimized Profit: ${best_profit:.2f}".ljust(59) + "║")
    print(f"║  Final Backtest PnL: ${final_pnl:.2f}".ljust(59) + "║")
    print("╠" + "═"*58 + "╣")
    print(f"║  Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}".ljust(59) + "║")
    print(f"║  Max Drawdown: {metrics['Max Drawdown']:.2%}".ljust(59) + "║")
    print("╚" + "═"*58 + "╝")
    print()


def main():
    """
    Main function to run the PyPairs trading pipeline.

    Pipeline Flow:
    1. DataEngine: Download data, find best correlated pair, calculate Z-Score
    2. Optimizer: Test multiple thresholds to find optimal entry point
    3. Backtester: Run final backtest with the best threshold
    4. RiskMetrics: Calculate Sharpe Ratio and Max Drawdown
    5. Visualizer: Display 3-panel dashboard
    """
    # =========================================================================
    # Configuration - Adjust these parameters as needed
    # =========================================================================
    TICKERS = ['F', 'GM', 'TM', 'HMC']  # Ford, GM, Toyota, Honda (auto sector)
    PERIOD = '1y'                        # Data period (1y = 1 year)
    EXIT_THRESHOLD = 0.0                 # Z-Score level to exit positions

    print("\n" + "🚗"*30)
    print("       PYPAIRS - Statistical Arbitrage System")
    print("       Pairs Trading with Z-Score Strategy")
    print("🚗"*30)

    # =========================================================================
    # Stage 1: Data Ingestion & Pair Selection
    # =========================================================================
    print_header(1, "Data Ingestion & Pair Selection")

    try:
        engine = DataEngine(verbose=True)
        result_df, pair_info = engine.run_full_pipeline(TICKERS, period=PERIOD)
        ticker_a, ticker_b, correlation = pair_info
        print(f"\n✓ Best pair found: {ticker_a} & {ticker_b}")
        print(f"✓ Correlation: {correlation:.4f}")
        print(f"✓ Data points: {len(result_df)}")
    except Exception as e:
        print(f"\n✗ Error in data ingestion: {e}")
        sys.exit(1)

    # =========================================================================
    # Stage 2: Strategy Optimization
    # =========================================================================
    print_header(2, "Strategy Optimization")

    try:
        optimizer = StrategyOptimizer()
        optimization_df = optimizer.optimize_thresholds(Backtester, result_df)

        # Find the best threshold (row with maximum profit)
        best_idx = optimization_df['Final Profit'].idxmax()
        best_threshold = optimization_df.loc[best_idx, 'Threshold']
        best_profit = optimization_df.loc[best_idx, 'Final Profit']

        print(f"\n✓ Thresholds tested: {len(optimization_df)}")
        print(f"✓ Best threshold: {best_threshold}")
        print(f"✓ Best profit: ${best_profit:.2f}")

        # Show optimization results table
        print("\nOptimization Grid Results:")
        print("-" * 30)
        for _, row in optimization_df.iterrows():
            marker = " ◄── BEST" if row['Threshold'] == best_threshold else ""
            print(f"  Threshold {row['Threshold']:.1f}: ${row['Final Profit']:>8.2f}{marker}")
    except Exception as e:
        print(f"\n✗ Error in optimization: {e}")
        sys.exit(1)

    # =========================================================================
    # Stage 3: Final Backtest with Optimal Threshold
    # =========================================================================
    print_header(3, "Final Backtest")

    try:
        backtester = Backtester()
        backtest_df = backtester.run_backtest(
            result_df,
            entry_threshold=best_threshold,
            exit_threshold=EXIT_THRESHOLD
        )

        final_pnl = backtest_df['Cumulative PnL'].iloc[-1]
        total_trades = (backtest_df['Position'].diff() != 0).sum()

        print("\n✓ Backtest completed")
        print(f"✓ Entry threshold: {best_threshold}")
        print(f"✓ Exit threshold: {EXIT_THRESHOLD}")
        print(f"✓ Final PnL: ${final_pnl:.2f}")
        print(f"✓ Position changes: {total_trades}")
    except Exception as e:
        print(f"\n✗ Error in backtest: {e}")
        sys.exit(1)

    # =========================================================================
    # Stage 4: Risk Analysis
    # =========================================================================
    print_header(4, "Risk Analysis")

    try:
        risk_engine = RiskMetrics()
        metrics = risk_engine.calculate_metrics(backtest_df['Cumulative PnL'])

        print("\n✓ Risk metrics calculated")
        print(f"✓ Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
        print(f"✓ Max Drawdown: {metrics['Max Drawdown']:.2%}")

        # Interpret the results
        sharpe = metrics['Sharpe Ratio']
        if sharpe > 1.0:
            print("  → Sharpe > 1.0: Good risk-adjusted returns")
        elif sharpe > 0.5:
            print("  → Sharpe 0.5-1.0: Moderate risk-adjusted returns")
        else:
            print("  → Sharpe < 0.5: Poor risk-adjusted returns")

        drawdown = metrics['Max Drawdown']
        if drawdown > -0.10:
            print("  → Max Drawdown < 10%: Low risk strategy")
        elif drawdown > -0.20:
            print("  → Max Drawdown 10-20%: Moderate risk strategy")
        else:
            print("  → Max Drawdown > 20%: High risk strategy")
    except Exception as e:
        print(f"\n✗ Error in risk analysis: {e}")
        sys.exit(1)

    # =========================================================================
    # Stage 5: Visualization
    # =========================================================================
    print_header(5, "Dashboard Visualization")

    # Print final summary before showing the dashboard
    print_summary(pair_info, best_threshold, best_profit, final_pnl, metrics)

    try:
        visualizer = Visualizer()
        print("Generating dashboard...")

        # Show the dashboard (optionally save to file)
        visualizer.plot_dashboard(
            backtest_df=backtest_df,
            optimization_df=optimization_df,
            pair_names=(ticker_a, ticker_b),
            entry_threshold=best_threshold,
            save_path=None  # Set to 'dashboard.png' to save instead of display
        )

        print("✓ Dashboard displayed successfully")
    except Exception as e:
        print(f"\n✗ Error in visualization: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("  PyPairs pipeline completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()