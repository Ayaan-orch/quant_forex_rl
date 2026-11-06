import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class AdvancedForexEnv(gym.Env):
    """
    Multi-Asset Institutional H4 Pullback Gatekeeper Environment.
    
    Features:
    - Multi-asset execution (USD/JPY, EUR/USD, GBP/USD) on H4 timeframe
    - Low friction drag (< 3.0% of ATR on H4 vs 17% on H1)
    - Macro Daily 200-EMA and Daily RSI confluence (zero lookahead)
    - Dynamic session-based variable spreads and slippage
    - 2.5 : 1 Asymmetric Reward-to-Risk bracket with breakeven & trailing stops
    """
    metadata = {'render_modes': ['human']}

    def __init__(
        self, 
        df: pd.DataFrame, 
        initial_balance: float = 10000.0, 
        enable_slippage: bool = True,
        random_start: bool = False
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.enable_slippage = enable_slippage
        self.random_start = random_start
        
        # Action 0: Pass (Reject candidate), Action 1: Execute (Accept candidate)
        self.action_space = spaces.Discrete(2)
        
        # 11-dimensional observation state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
        self.current_step = 25
        self.candidate_direction = 0

    def _get_dynamic_friction(self, entry_price: float, row: pd.Series) -> float:
        """Calculates realistic basis-point friction scaled by session liquidity."""
        session_factor = float(row.get('session_factor', 1.0))
        # Base spread ~ 1.5 bps (0.015%), tighter during London/NY
        base_bps = 0.00015 / max(session_factor, 0.8)
        # Commission: $3.50/lot ~ 0.00006 (0.6 bps)
        comm_bps = 0.00006
        # Stochastic slippage
        slip_bps = abs(np.random.normal(0.0, 0.00004)) if self.enable_slippage else 0.0
        return entry_price * (base_bps + comm_bps + slip_bps)

    def _seek_next_setup(self) -> bool:
        """Scans forward to find high-conviction macro H4 pullbacks."""
        max_idx = len(self.df) - 25
        while self.current_step < max_idx:
            row = self.df.iloc[self.current_step]
            trend = row.get('trend_spread', 0.0)
            rsi_val = row.get('norm_rsi', 0.0)
            atr_val = row.get('norm_atr', 0.0)
            htf_trend = row.get('htf_trend_200', 0.0)

            # Volatility floor on H4 (0.15% minimum candle range)
            if atr_val >= 0.0015:
                # Bullish Macro Pullback: H4 Uptrend, healthy RSI pullback (35-52), Daily 200 EMA confirmation
                if trend >= 0.0004 and (-0.30 <= rsi_val <= 0.05) and htf_trend >= 0.0:
                    self.candidate_direction = 1
                    return False

                # Bearish Macro Pullback: H4 Downtrend, healthy RSI pullback upward (48-65), Daily 200 EMA confirmation
                elif trend <= -0.0004 and (-0.05 <= rsi_val <= 0.30) and htf_trend <= 0.0:
                    self.candidate_direction = -1
                    return False

            self.current_step += 1
        return True

    def _get_obs(self) -> np.ndarray:
        safe_idx = min(self.current_step, len(self.df) - 1)
        row = self.df.iloc[safe_idx]
        obs = np.array([
            row.get('log_ret', 0.0),
            row.get('norm_rsi', 0.0),
            row.get('norm_atr', 0.0),
            row.get('norm_vol', 0.0),
            row.get('trend_spread', 0.0),
            row.get('sentiment_score', 0.0),
            row.get('event_surprise', 0.0),
            row.get('time_to_event', 1.0),
            row.get('htf_trend_200', 0.0),
            row.get('htf_rsi', 0.0),
            float(self.candidate_direction)
        ], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=2.0, neginf=-2.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        use_random = self.random_start
        if options and "random_start" in options:
            use_random = options["random_start"]

        if use_random and len(self.df) > 200:
            max_start = max(26, len(self.df) - 250)
            self.current_step = self.np_random.integers(25, max_start)
        else:
            self.current_step = 25

        self._seek_next_setup()
        return self._get_obs(), {}

    def step(self, action: int):
        entry_idx = self.current_step
        row_entry = self.df.iloc[entry_idx]
        entry_price = float(row_entry['Close'])
        norm_atr_val = float(row_entry['norm_atr'])
        raw_atr = max(norm_atr_val * entry_price, entry_price * 0.0015)
        direction = self.candidate_direction

        # Microstructure costs
        total_friction = self._get_dynamic_friction(entry_price, row_entry)

        # 2.5 : 1 Target Asymmetry on H4
        if direction == 1:
            take_profit_price = entry_price + (2.5 * raw_atr)
            stop_loss_price = entry_price - (1.0 * raw_atr)
        else:
            take_profit_price = entry_price - (2.5 * raw_atr)
            stop_loss_price = entry_price + (1.0 * raw_atr)

        trade_closed = False
        bar_offset = 1
        max_bars = 18  # 18 * 4h = 72 hours (3 trading days max)
        overnight_swap = 0.0
        exit_price = entry_price

        while not trade_closed and (entry_idx + bar_offset < len(self.df)) and (bar_offset <= max_bars):
            bar = self.df.iloc[entry_idx + bar_offset]
            high = float(bar['High'])
            low = float(bar['Low'])

            # 6 H4 bars = 24 hours (daily swap)
            if bar_offset % 6 == 0:
                overnight_swap += entry_price * 0.000015

            if direction == 1:
                # 1. Check exit against stop/target from prior bar
                if low <= stop_loss_price:
                    exit_price = stop_loss_price
                    trade_closed = True
                elif high >= take_profit_price:
                    exit_price = take_profit_price
                    trade_closed = True
                else:
                    # 2. Update dynamic trailing stop for subsequent bars
                    # Breakeven migration at +1.0 ATR
                    if (high - entry_price) >= (1.0 * raw_atr):
                        stop_loss_price = max(stop_loss_price, entry_price + total_friction)
                    # Trailing stop at +1.6 ATR
                    if (high - entry_price) >= (1.6 * raw_atr):
                        trailing_sl = high - (0.85 * raw_atr)
                        stop_loss_price = max(stop_loss_price, trailing_sl)
            else:
                if high >= stop_loss_price:
                    exit_price = stop_loss_price
                    trade_closed = True
                elif low <= take_profit_price:
                    exit_price = take_profit_price
                    trade_closed = True
                else:
                    if (entry_price - low) >= (1.0 * raw_atr):
                        stop_loss_price = min(stop_loss_price, entry_price - total_friction)
                    if (entry_price - low) >= (1.6 * raw_atr):
                        trailing_sl = low + (0.85 * raw_atr)
                        stop_loss_price = min(stop_loss_price, trailing_sl)

            bar_offset += 1

        # Timeout exit at bar close
        if not trade_closed:
            exit_idx = min(entry_idx + bar_offset, len(self.df) - 1)
            exit_price = float(self.df.iloc[exit_idx]['Close'])

        total_costs = total_friction + overnight_swap
        gross = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
        step_return = (gross - total_costs) / entry_price

        # Asymmetric Signal-Detection Reward Formulation
        norm_r = float(step_return / (raw_atr / entry_price))
        if action == 1:
            if step_return > 0:
                reward = float(norm_r * 2.2)
            else:
                reward = float(norm_r * 1.0)
        else:
            # Action 0: Fold / Pass
            if step_return > 0:
                reward = float(-norm_r * 1.0) # Opportunity cost
            else:
                reward = float(abs(norm_r) * 0.3) # Disciplined pass

        if action == 0:
            step_return = 0.0

        self.current_step = entry_idx + (bar_offset if action == 1 else 1)
        if self.current_step >= len(self.df) - 25:
            terminated = True
        else:
            terminated = self._seek_next_setup()

        info = {
            "step": self.current_step,
            "step_return": step_return,
            "action": action,
            "direction": direction if action == 1 else 0,
            "friction_paid": total_costs if action == 1 else 0.0
        }
        return self._get_obs(), reward, terminated, False, info

if __name__ == "__main__":
    from pathlib import Path
    from stable_baselines3.common.env_checker import check_env

    data_path = Path(__file__).resolve().parent.parent / "data" / "processed_market_features.csv"
    test_df = pd.read_csv(data_path)
    env = AdvancedForexEnv(test_df)
    print("\nValidating multi-asset H4 environment...")
    check_env(env, warn=True)
    print("Multi-asset H4 environment passed all gymnasium and stable-baselines3 checks!")
