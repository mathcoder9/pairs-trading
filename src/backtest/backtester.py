from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from .data_utils import clean_data, check_data_valid
import matplotlib.pyplot as plt


@dataclass
class BacktestParameters:
    """Parameters for backtest configuration"""

    initial_capital: float
    trading_fee: float  # Percentage as decimal (e.g., 0.001 for 0.1%)
    slippage: float  # Percentage as decimal
    interest_rate: float  # Annual interest rate for unused cash
    risk_free_rate: float  # Annual risk-free rate for Sharpe calculation
    position_size: float  # Percentage of capital to use per trade
    stop_loss: float = 0.03  # Stop Loss threshold
    cool_off: int = 3  # Cool off period after stop loss is triggered in days
    date_column: str = "CloseTime"
    pair_column: str = "Pair"
    price_column: str = "ClosePrice"


class Strategy(ABC):
    """Abstract base class for trading strategies"""

    def __init__(self, ticker1: str, ticker2: str):
        self.ticker1 = ticker1
        self.ticker2 = ticker2

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals and return a series of dates and corresponding position that should be held

        Parameters:
        - data: DataFrame with columns [date, ticker, price]

        Returns:
        - Series with date index and values:
          1 for long t1/short t2
          -1 for short t1/long t2
          0 for no position
        """
        pass

    @abstractmethod
    def get_units(
        self, price1: float, price2: float, position_size: float, is_buy: bool
    ):
        """
        Calculate unit1 and unit2, the amount of ticker1 and ticker2 bought respectively
        (>0 for long, <0 for short)

        Parameters:
        - price1: Price of ticker1
        - price2: Price of ticker2
        - position_size: Amount of cash to spend on position

        Returns:
        - (unit1, unit2)
        """
        pass


class PairPosition:
    """Tracks a single pair trading position"""

    def __init__(
        self,
        ticker1: str,
        ticker2: str,
        entry_price1: float,
        entry_price2: float,
        units1: float,
        units2: float,
        position_type: int,
        entry_date: datetime,
        initial_position_value: float,
    ):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.entry_price1 = entry_price1
        self.entry_price2 = entry_price2
        self.units1 = units1
        self.units2 = units2
        self.position_type = position_type
        self.entry_date = entry_date
        self.initial_position_value = initial_position_value

        # Exit information
        self.exit_price1: Optional[float] = None
        self.exit_price2: Optional[float] = None
        self.exit_date: Optional[datetime] = None
        self.pnl: Optional[float] = None

    def calculate_pnl(self, exit_price1: float, exit_price2: float) -> float:
        """Calculate PnL"""
        pnl1 = self.units1 * (exit_price1 - self.entry_price1)
        pnl2 = self.units2 * (exit_price2 - self.entry_price2)
        return pnl1 + pnl2

    def pnl_as_fraction_of_initial(
        self, exit_price1: float, exit_price2: float
    ) -> float:
        """Calculate PnL as percentage of initial position value"""
        return (
            self.calculate_pnl(exit_price1, exit_price2) / self.initial_position_value
        )


class Backtest:
    """Basic backtester for pairs trading"""

    def __init__(self, params: BacktestParameters):
        self._params = params
        self._cash = self._params.initial_capital
        self._current_position: Optional[PairPosition] = None
        self._closed_positions: List[PairPosition] = []
        self._performance_history: List[Dict] = []
        self._reset()

    def _reset(self) -> None:
        """Reset backtester state"""
        self._cash = self._params.initial_capital
        self._current_position = None
        self._closed_positions: List[PairPosition] = []
        self._performance_history: List[Dict] = []

    def _apply_trading_costs(self, price: float, is_buy: bool) -> float:
        """Apply trading fees and slippage to price"""
        cost_multiplier = 1 + (self._params.trading_fee + self._params.slippage) * (
            1 if is_buy else -1
        )
        return price * cost_multiplier

    def _open_position(
        self,
        ticker1: float,
        ticker2: float,
        price1: float,
        price2: float,
        signal: int,
        strategy: Strategy,
        date: datetime,
    ) -> None:
        """Opens a new position"""
        initial_position_value = self._cash * self._params.position_size
        entry_price1 = self._apply_trading_costs(price1, signal > 0)
        entry_price2 = self._apply_trading_costs(price2, signal < 0)
        unit1, unit2 = strategy.get_units(
            entry_price1, entry_price2, initial_position_value, signal > 0
        )
        # Open new position
        self._current_position = PairPosition(
            ticker1=ticker1,
            ticker2=ticker2,
            entry_price1=entry_price1,
            entry_price2=entry_price2,
            units1=unit1,
            units2=unit2,
            position_type=signal,
            entry_date=date,
            initial_position_value=initial_position_value,
        )
        self._cash -= initial_position_value

    def _exit_position(
        self,
        price1: float,
        price2: float,
        date: datetime,
        stop_loss_triggered: bool = False,
    ) -> None:
        """Exits current position"""
        if not self._current_position:
            raise Exception("There is no current position to close")

        exit_price1 = self._apply_trading_costs(
            price1, self._current_position.units1 < 0
        )
        exit_price2 = self._apply_trading_costs(
            price2, self._current_position.units2 < 0
        )

        # Calculate PnL and update position
        if stop_loss_triggered:
            print("SL Triggered:", date)
            pnl = (
                -1
                * self._current_position.initial_position_value
                * self._params.stop_loss
            )
        else:
            pnl = self._current_position.calculate_pnl(exit_price1, exit_price2)
        self._current_position.pnl = pnl
        self._current_position.exit_price1 = exit_price1
        self._current_position.exit_price2 = exit_price2
        self._current_position.exit_date = date

        # Update portfolio
        self._closed_positions.append(self._current_position)
        self._cash += self._current_position.initial_position_value + pnl
        self._current_position = None

    def _check_stop_loss(self, price1: float, price2: float) -> bool:
        """Checks if stop loss should be triggered"""
        if not self._current_position:
            raise Exception("There is no current position to check")

        exit_price1 = self._apply_trading_costs(
            price1, self._current_position.units1 < 0
        )
        exit_price2 = self._apply_trading_costs(
            price2, self._current_position.units2 < 0
        )
        returns = self._current_position.pnl_as_fraction_of_initial(
            exit_price1, exit_price2
        )
        return returns < -1 * self._params.stop_loss

    def _update_performance_history(
        self, price1: float, price2: float, date: datetime
    ) -> None:
        """Update the daily portfolio value record"""
        portfolio_value = self._cash
        if self._current_position:
            portfolio_value += (
                self._current_position.initial_position_value
                + self._current_position.calculate_pnl(price1, price2)
            )

        self._performance_history.append(
            {self._params.date_column: date, "portfolio_value": portfolio_value}
        )

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self._performance_history:
            return {}

        portfolio_values = pd.Series(
            [x["portfolio_value"] for x in self._performance_history],
            index=[x[self._params.date_column] for x in self._performance_history],
        )

        daily_returns = portfolio_values.pct_change().dropna()
        total_days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        total_return = (
            portfolio_values.iloc[-1] - portfolio_values.iloc[0]
        ) / portfolio_values.iloc[0]

        # Annualized metrics
        annualized_return = (1 + total_return) ** (365 / total_days) - 1
        excess_returns = (
            daily_returns - (1 + self._params.risk_free_rate) ** (1 / 365) + 1
        )
        sharpe_ratio = (
            np.sqrt(365) * (excess_returns.mean() / excess_returns.std())
            if len(excess_returns) > 1
            else 0
        )

        # Drawdown
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Trading statistics
        total_trades = len(self._closed_positions)
        winning_trades = sum(1 for pos in self._closed_positions if pos.pnl > 0)

        return {
            "total_return": total_return * 100,
            "annualized_return": annualized_return * 100,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown * 100,
            "win_rate": (winning_trades / total_trades * 100)
            if total_trades > 0
            else -1,
            "total_trades": total_trades,
            "final_portfolio_value": portfolio_values.iloc[-1],
        }

    def _plot_performance(self) -> None:
        """Plot portfolio value"""
        if not self._performance_history:
            return
        portfolio_values = pd.Series(
            [x["portfolio_value"] for x in self._performance_history],
            index=[x[self._params.date_column] for x in self._performance_history],
        )

        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_values)
        plt.title(f"Portfolio Value over time")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.show()

    def _print_backtest_summary(self):
        """Print summary of backtest"""
        stats = self._calculate_metrics()
        # Define formatting
        header = "Backtest Summary"
        separator = "─" * 50

        metrics = [
            ("Initial Portfolio Value", f"${self._params.initial_capital:,.2f}"),
            ("Final Portfolio Value", f"${stats['final_portfolio_value']:,.2f}"),
            ("Total Return", f"{stats['total_return']:,.2f}%"),
            ("Annualized Return", f"{stats['annualized_return']:,.2f}%"),
            ("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}"),
            ("Maximum Drawdown", f"{stats['max_drawdown']:.2f}%"),
            (
                "Win Rate",
                f"No Closed Positions"
                if stats["win_rate"] == -1
                else f"{stats['win_rate']:.1f}%",
            ),
            ("# of Closed Positions", f"{stats['total_trades']:,}"),
            ("# of Open Positions", "1" if self._current_position else "0"),
        ]

        # Calculate maximum label length for alignment
        max_label_length = max(len(label) for label, _ in metrics)

        # Print formatted summary
        print(f"\n{header}")
        print(separator)

        for label, value in metrics:
            # Right-align labels and left-align values
            print(f"{label:<{max_label_length}} : {value}")

        print(separator)

    def run_backtest(
        self,
        backtest_data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        strategy: Strategy,
        plot_performance: bool = True,
    ) -> None:
        """
        Run backtest using pre-generated signals

        Parameters:
        - backtest_data: DataFrame with columns [date, ticker, price]
        - start_date: Start of backtest period
        - end_date: End of backtest period
        - strategy: An implementation of Strategy
        """
        self._reset()

        ticker1 = strategy.ticker1
        ticker2 = strategy.ticker2

        data = backtest_data.copy()
        data = clean_data(
            data,
            start_date,
            end_date,
            ticker1,
            ticker2,
            self._params.date_column,
            self._params.pair_column,
        )
        if not check_data_valid(
            data,
            start_date,
            end_date,
            self._params.date_column,
            self._params.pair_column,
            self._params.price_column,
        ):
            print("Aborting....")
            return None

        signals, _ = strategy.generate_signals(data)
        data = data.sort_values(self._params.date_column)

        # Create price lookup for each asset
        prices = data.pivot(
            index=self._params.date_column,
            columns=self._params.pair_column,
            values=self._params.price_column,
        )

        skips = 0

        for date, signal in signals.items():
            if date not in prices.index:
                continue

            do_not_open_new_positions = False
            if skips > 0:
                skips -= 1
                do_not_open_new_positions = True

            # Get current prices
            price1 = prices.loc[date, ticker1]
            price2 = prices.loc[date, ticker2]

            # Apply daily interest to cash
            self._cash *= (1 + self._params.interest_rate) ** (1 / 365)

            # Check if exit signal is given
            if (
                self._current_position
                and signal != self._current_position.position_type
            ):
                self._exit_position(price1, price2, date)

            # Check if stop loss is triggered.
            if self._current_position and self._check_stop_loss(price1, price2):
                self._exit_position(price1, price2, date, True)
                skips = self._params.cool_off
                do_not_open_new_positions = True

            # Handle new position entry
            if (
                not do_not_open_new_positions
                and not self._current_position
                and signal in [1, -1]
                and self._cash > 0
            ):
                self._open_position(
                    ticker1, ticker2, price1, price2, signal, strategy, date
                )

            # Record daily performance
            self._update_performance_history(price1, price2, date)

        self._print_backtest_summary()
        if plot_performance:
            self._plot_performance()


class PairsWithConstantStrategy(Strategy):
    """Pairs Trading Strategy"""

    def __init__(
        self,
        ticker1: str,
        ticker2: str,
        std: float,
        buy_entry: float,
        buy_exit: float,
        sell_entry: float,
        sell_exit: float,
        constant: float,
        hedge_ratio: float,
    ):
        super().__init__(ticker1, ticker2)
        self._std = std
        self._buy_entry = buy_entry
        self._buy_exit = buy_exit
        self._sell_entry = sell_entry
        self._sell_exit = sell_exit
        self._constant = constant
        self._hedge_ratio = hedge_ratio

    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Generate all signals including stops"""
        # Calculate price ratios
        pair_data = data.pivot(index="CloseTime", columns="Pair", values="ClosePrice")

        p1 = pair_data.loc[:, self.ticker1]
        p2 = pair_data.loc[:, self.ticker2]

        spread = p1 - p2 * self._hedge_ratio - self._constant

        signals = pd.Series(index=spread.index)
        previous_state = 0  # -1 for short, 1 for long, 0 for no position
        for i in range(len(spread)):
            signals.iloc[i] = previous_state
            if previous_state == 0:
                if spread.iloc[i] < -self._buy_entry * self._std:
                    signals.iloc[i] = 1
                elif spread.iloc[i] > self._sell_entry * self._std:
                    signals.iloc[i] = -1
            elif previous_state == 1:
                if spread.iloc[i] > self._sell_entry * self._std:
                    signals.iloc[i] = -1
                elif spread.iloc[i] > -self._buy_exit * self._std:
                    signals.iloc[i] = 0
            elif previous_state == -1:
                if spread.iloc[i] < -self._buy_entry * self._std:
                    signals.iloc[i] = 1
                elif spread.iloc[i] < self._sell_exit * self._std:
                    signals.iloc[i] = 0
            previous_state = signals.iloc[i]

        return signals, spread

    def display_signal(self, data: pd.DataFrame) -> None:
        """Plot signal and spread"""
        signals, spread = self.generate_signals(data)
        plt.figure(figsize=(10, 6))
        plt.plot(spread)
        plt.plot(signals * self._std * 3)
        plt.axhline(0, color="red", linestyle="--")
        plt.axhline(self._sell_entry * self._std, color="blue", linestyle="--")
        plt.axhline(-self._buy_entry * self._std, color="blue", linestyle="--")
        plt.title(f"Spread of {self.ticker1} hedged by {self.ticker2}")
        plt.xlabel("Date")
        plt.ylabel("Residuals")
        plt.show()

    def get_units(
        self, price1: float, price2: float, position_size: float, is_buy: bool
    ) -> Tuple[float, float]:
        """For a given pair of prices, position_size, and trade direction calculate the units of each asset bought"""
        position_unit_cost = price1 + price2 * self._hedge_ratio
        total_units = position_size / position_unit_cost
        return (
            (total_units, -1 * total_units * self._hedge_ratio)
            if is_buy
            else (-1 * total_units, total_units * self._hedge_ratio)
        )


class BuyAndHoldStrategy(Strategy):
    """Buy and Hold Trading Strategy"""

    def __init__(
        self,
        ticker1: str,
    ):
        super().__init__(
            ticker1, "ETHUSDT"
        )  # ETHUSDT is used as a dummy variable so that the Backtest class can be used

    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Generate signals"""
        # Calculate price ratios
        pair_data = data.pivot(index="CloseTime", columns="Pair", values="ClosePrice")
        p1 = pair_data.loc[:, self.ticker1]
        signals = p1 / p1
        return signals, None

    def get_units(
        self, price1: float, price2: float, position_size: float, is_buy: bool
    ):
        """For a given pair of prices, position_size, and trade direction calculate the units of each asset bought"""
        position_unit_cost = price1
        total_units = position_size / position_unit_cost
        return (total_units, 0) if is_buy else (-1 * total_units, 0)
