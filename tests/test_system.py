import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from environment import AdvancedForexEnv
from multi_asset_execution import check_currency_correlation_limit, is_weekend_lockout


class TestQuantForexSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create synthetic H4 OHLCV dataframe for invariant tests
        n_bars = 120
        np.random.seed(42)
        base_price = 1.1000
        returns = np.random.normal(0, 0.002, n_bars)
        prices = base_price * np.exp(np.cumsum(returns))
        
        dates = pd.date_range("2026-01-01", periods=n_bars, freq="4h")
        cls.test_df = pd.DataFrame({
            'time': dates,
            'Open': prices * (1 + np.random.normal(0, 0.0005, n_bars)),
            'High': prices * (1 + abs(np.random.normal(0, 0.001, n_bars))),
            'Low': prices * (1 - abs(np.random.normal(0, 0.001, n_bars))),
            'Close': prices,
            'Volume': np.random.randint(100, 1000, n_bars),
            'log_ret': np.insert(np.diff(np.log(prices)), 0, 0.0),
            'norm_rsi': np.random.uniform(-0.5, 0.5, n_bars),
            'norm_atr': np.random.uniform(0.0015, 0.0040, n_bars),
            'norm_vol': np.random.uniform(-1.0, 1.0, n_bars),
            'trend_spread': np.random.uniform(-0.002, 0.002, n_bars),
            'sentiment_score': np.random.uniform(-0.5, 0.5, n_bars),
            'event_surprise': np.zeros(n_bars),
            'time_to_event': np.ones(n_bars),
            'htf_trend_200': np.random.uniform(-0.01, 0.01, n_bars),
            'htf_rsi': np.random.uniform(-0.3, 0.3, n_bars),
            'candidate_direction': np.ones(n_bars)
        })

    def test_environment_gym_contract(self):
        """Verifies AdvancedForexEnv fulfills Gymnasium API contracts."""
        env = AdvancedForexEnv(self.test_df, random_start=False)
        obs, info = env.reset()
        
        self.assertEqual(obs.shape, (11,), "Observation vector must be 11-dimensional.")
        self.assertFalse(np.isnan(obs).any(), "Observation vector must contain no NaN values.")
        self.assertFalse(np.isinf(obs).any(), "Observation vector must contain no Inf values.")

        # Test step mechanics
        obs, reward, terminated, truncated, info = env.step(1)
        self.assertIsInstance(reward, (float, np.floating))
        self.assertFalse(np.isnan(reward), "Reward must not be NaN.")

    def test_currency_correlation_guard(self):
        """Verifies correlation guard prevents stacking correlated USD exposure."""
        # Mock active position: Long EURUSD (which is Short USD)
        active_positions = [
            {"symbol": "EURUSD", "direction": 1}  # Long EURUSD = Short USD
        ]
        
        # New proposal: Long GBPUSD (which is also Short USD)
        # Should be rejected because it doubles Short USD exposure
        allowed, reason = check_currency_correlation_limit("GBPUSD", 1, active_positions, max_usd_exposure=1)
        self.assertFalse(allowed, "Correlated GBPUSD Long should be rejected when EURUSD Long is already open.")
        self.assertIn("USD", reason)

        # Counter proposal: Short USDJPY (which is also Short USD)
        allowed, _ = check_currency_correlation_limit("USDJPY", -1, active_positions, max_usd_exposure=1)
        self.assertFalse(allowed, "USDJPY Short should be blocked when USD short limit is reached.")

        # Non-conflicting proposal: Long USDJPY (which is Long USD, balancing exposure)
        allowed, _ = check_currency_correlation_limit("USDJPY", 1, active_positions, max_usd_exposure=1)
        self.assertTrue(allowed, "USDJPY Long should be allowed as it balances USD direction.")

    def test_weekend_lockout_detection(self):
        """Verifies Friday evening weekend lockout detection."""
        from datetime import datetime, timezone
        
        # Friday 20:00 UTC (Market closing soon -> Should lockout)
        friday_night = datetime(2026, 11, 6, 20, 0, tzinfo=timezone.utc)
        self.assertTrue(is_weekend_lockout(friday_night), "Friday after 19:00 UTC must trigger lockout.")
        
        # Wednesday 14:00 UTC (Active mid-week trading -> Should NOT lockout)
        wednesday_noon = datetime(2026, 11, 4, 14, 0, tzinfo=timezone.utc)
        self.assertFalse(is_weekend_lockout(wednesday_noon), "Midweek session must not trigger lockout.")

    def test_economic_news_shield(self):
        """Verifies that high-impact economic news windows trigger execution lockout."""
        from datetime import datetime, timezone
        from news_filter import EconomicNewsFilter

        # 1st Friday of month at 13:00 UTC -> NFP window -> Must trigger shield
        nfp_release_time = datetime(2026, 11, 6, 13, 0, tzinfo=timezone.utc)
        is_active, reason = EconomicNewsFilter.check_high_impact_window(nfp_release_time)
        self.assertTrue(is_active, "NFP release time must activate news shield.")
        self.assertIn("NFP", reason)

        # Normal trading hour -> Should be clear
        calm_time = datetime(2026, 11, 10, 10, 0, tzinfo=timezone.utc)
        is_active, reason = EconomicNewsFilter.check_high_impact_window(calm_time)
        self.assertFalse(is_active, "Normal session should not activate news shield.")
        self.assertEqual(reason, "CLEAR")

    def test_pretrained_policy_inference(self):
        """Verifies ppo_forex_policy.zip loads and makes valid predictions."""
        from sb3_contrib import RecurrentPPO
        from stable_baselines3 import PPO

        model_path = ROOT_DIR / "ppo_forex_policy"
        if (ROOT_DIR / "ppo_forex_policy.zip").exists():
            try:
                model = RecurrentPPO.load(str(model_path))
            except Exception:
                model = PPO.load(str(model_path))
            
            dummy_obs = np.zeros(11, dtype=np.float32)
            action, _ = model.predict(dummy_obs, deterministic=True)
            self.assertIn(int(action), [0, 1], "Model action must be binary (0 or 1).")

    def test_monte_carlo_resampling(self):
        """Verifies Monte Carlo resampling statistical integrity."""
        from stress_test import run_monte_carlo_stress_test

        synthetic_returns = np.array([0.02, -0.01, 0.025, -0.008, 0.015, -0.005, 0.03, -0.01])
        results = run_monte_carlo_stress_test(synthetic_returns, n_simulations=100)

        self.assertIn("var_95_dd_pct", results)
        self.assertIn("prob_profit", results)
        self.assertGreaterEqual(results["prob_profit"], 0.0)
        self.assertLessEqual(results["prob_profit"], 100.0)
        self.assertEqual(results["equity_paths"].shape, (100, len(synthetic_returns) + 1))


if __name__ == "__main__":
    unittest.main()
