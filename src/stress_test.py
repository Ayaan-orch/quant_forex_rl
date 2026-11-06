"""
Institutional Monte Carlo Stress-Testing & Drawdown Robustness Engine.

Performs 2,500 stochastic permutations with bootstrap resampling and adverse slippage injection
to evaluate worst-case Value at Risk (VaR), Conditional VaR, and ruin probabilities.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from backtest import load_policy, MODEL_PATH, DATA_PATH
from environment import AdvancedForexEnv
import torch as th


def extract_backtest_trade_returns() -> np.ndarray:
    """Runs out-of-sample backtest to extract per-trade returns."""
    df = pd.read_csv(DATA_PATH)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    env = AdvancedForexEnv(test_df, random_start=False)
    model, is_recurrent = load_policy(MODEL_PATH)
    
    obs, _ = env.reset()
    terminated = False
    trade_returns = []

    if is_recurrent:
        state = (np.zeros(model.policy.lstm_hidden_state_shape), np.zeros(model.policy.lstm_hidden_state_shape))
    episode_starts = np.array([True])

    while not terminated:
        if is_recurrent:
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            with th.no_grad():
                states = (th.tensor(state[0], dtype=th.float32), th.tensor(state[1], dtype=th.float32))
                ep_tensor = th.tensor(episode_starts, dtype=th.float32)
                dist, new_states = model.policy.get_distribution(obs_tensor, states, ep_tensor)
                p_exec = float(dist.distribution.probs.cpu().numpy()[0][1])
                state = (new_states[0].cpu().numpy(), new_states[1].cpu().numpy())
            episode_starts = np.array([False])
            action = 1 if p_exec >= 0.88 else 0
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(int(action))
        if info.get("action") == 1:
            trade_returns.append(info["step_return"])

    return np.array(trade_returns)


def run_monte_carlo_stress_test(
    trade_returns: np.ndarray,
    n_simulations: int = 2500,
    seed: int = 42
) -> dict:
    """
    Simulates n_simulations trade paths with bootstrap resampling and random adverse friction.
    """
    np.random.seed(seed)
    n_trades = len(trade_returns)
    if n_trades == 0:
        raise ValueError("No trades provided for Monte Carlo analysis.")

    final_returns = []
    max_drawdowns = []
    equity_paths = np.zeros((n_simulations, n_trades + 1))
    equity_paths[:, 0] = 1.0

    for i in range(n_simulations):
        # Bootstrap resampling with replacement
        sample_indices = np.random.choice(n_trades, size=n_trades, replace=True)
        sampled_returns = trade_returns[sample_indices]

        # Inject stochastic adverse slippage noise (0.0 to 1.5 bps)
        adverse_slippage = np.random.uniform(0.0, 0.00015, size=n_trades)
        stressed_returns = sampled_returns - adverse_slippage

        # Compute equity path
        eq = np.cumprod(1.0 + stressed_returns)
        equity_paths[i, 1:] = eq

        final_returns.append(eq[-1] - 1.0)

        # Calculate max drawdown
        peaks = np.maximum.accumulate(eq)
        drawdowns = (eq - peaks) / peaks
        max_drawdowns.append(abs(float(np.min(drawdowns))))

    final_returns = np.array(final_returns)
    max_drawdowns = np.array(max_drawdowns)

    # Risk Metrics
    var_95_dd = float(np.percentile(max_drawdowns, 95))
    var_99_dd = float(np.percentile(max_drawdowns, 99))
    cvar_95_dd = float(np.mean(max_drawdowns[max_drawdowns >= var_95_dd]))

    prob_profit = float(np.mean(final_returns > 0.0) * 100)
    prob_dd_over_5pct = float(np.mean(max_drawdowns > 0.05) * 100)
    prob_dd_over_10pct = float(np.mean(max_drawdowns > 0.10) * 100)

    median_return = float(np.median(final_returns) * 100)
    return_5th = float(np.percentile(final_returns, 5) * 100)
    return_95th = float(np.percentile(final_returns, 95) * 100)

    results = {
        "n_simulations": n_simulations,
        "n_trades": n_trades,
        "prob_profit": prob_profit,
        "median_return_pct": median_return,
        "return_5th_pct": return_5th,
        "return_95th_pct": return_95th,
        "median_max_dd_pct": float(np.median(max_drawdowns) * 100),
        "var_95_dd_pct": var_95_dd * 100,
        "var_99_dd_pct": var_99_dd * 100,
        "cvar_95_dd_pct": cvar_95_dd * 100,
        "prob_dd_over_5pct": prob_dd_over_5pct,
        "prob_dd_over_10pct": prob_dd_over_10pct,
        "equity_paths": equity_paths,
        "max_drawdowns": max_drawdowns,
        "final_returns": final_returns
    }

    return results


def plot_stress_test_results(results: dict, output_path: Path):
    """Generates visual Monte Carlo path distribution and drawdown histogram."""
    equity_paths = results["equity_paths"]
    max_drawdowns = results["max_drawdowns"] * 100
    n_trades = results["n_trades"]

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=200)

    # Plot 1: 100 Representative Monte Carlo Equity Paths
    trade_steps = np.arange(n_trades + 1)
    for i in range(min(150, len(equity_paths))):
        ax1.plot(trade_steps, (equity_paths[i] - 1.0) * 100, color='#00d2ff', alpha=0.08, linewidth=1.0)

    # Median and Percentile bands
    p5 = np.percentile(equity_paths, 5, axis=0)
    p50 = np.percentile(equity_paths, 50, axis=0)
    p95 = np.percentile(equity_paths, 95, axis=0)

    ax1.plot(trade_steps, (p50 - 1.0) * 100, color='#00ff88', linewidth=2.5, label=f"Median Path (+{results['median_return_pct']:.2f}%)")
    ax1.plot(trade_steps, (p95 - 1.0) * 100, color='#38ef7d', linestyle='--', linewidth=1.5, label=f"95th Percentile (+{results['return_95th_pct']:.2f}%)")
    ax1.plot(trade_steps, (p5 - 1.0) * 100, color='#ff4b4b', linestyle='--', linewidth=1.5, label=f"5th Percentile ({results['return_5th_pct']:.2f}%)")

    ax1.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax1.set_title(f"Monte Carlo Equity Trajectories ({results['n_simulations']:,} Paths)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Executed Trades Sequence", fontsize=11)
    ax1.set_ylabel("Cumulative Strategy Return (%)", fontsize=11)
    ax1.grid(True, alpha=0.15)
    ax1.legend(loc='upper left', framealpha=0.3)

    # Plot 2: Max Drawdown Probability Distribution
    ax2.hist(max_drawdowns, bins=40, color='#ff5252', alpha=0.75, edgecolor='black', density=True)
    ax2.axvline(results['var_95_dd_pct'], color='#ffcc00', linestyle='--', linewidth=2.0, 
                label=f"95% VaR DD: -{results['var_95_dd_pct']:.2f}%")
    ax2.axvline(results['var_99_dd_pct'], color='#ff1744', linestyle='-', linewidth=2.2, 
                label=f"99% VaR DD: -{results['var_99_dd_pct']:.2f}%")
    ax2.axvline(results['cvar_95_dd_pct'], color='#d500f9', linestyle=':', linewidth=2.0, 
                label=f"95% CVaR DD: -{results['cvar_95_dd_pct']:.2f}%")

    ax2.set_title("Drawdown Distribution & Value at Risk (VaR)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Maximum Drawdown Magnitude (%)", fontsize=11)
    ax2.set_ylabel("Probability Density", fontsize=11)
    ax2.grid(True, alpha=0.15)
    ax2.legend(loc='upper right', framealpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[Stress Test] Saved distribution chart to {output_path}")


def main():
    print("=" * 65)
    print("  QUANT FOREX RL — INSTITUTIONAL MONTE CARLO STRESS TEST")
    print("=" * 65)

    print("\n[Step 1] Extracting Out-of-Sample trade returns...")
    trade_returns = extract_backtest_trade_returns()
    print(f"Extracted {len(trade_returns)} high-conviction trades.")

    print(f"\n[Step 2] Executing 2,500 Monte Carlo bootstrap paths with adverse friction...")
    results = run_monte_carlo_stress_test(trade_returns, n_simulations=2500)

    print("\n" + "=" * 65)
    print("  STRESS TEST STATISTICAL AUDIT REPORT")
    print("=" * 65)
    print(f"  • Probability of Profitable Outcome:     {results['prob_profit']:.1f}%")
    print(f"  • Median Cumulative Net Return:          +{results['median_return_pct']:.2f}%")
    print(f"  • 90% Confidence Interval:               [{results['return_5th_pct']:+.2f}%, {results['return_95th_pct']:+.2f}%]")
    print(f"  • Median Maximum Drawdown:               -{results['median_max_dd_pct']:.2f}%")
    print(f"  • 95% Value at Risk (VaR Drawdown):      -{results['var_95_dd_pct']:.2f}%")
    print(f"  • 99% Value at Risk (VaR Drawdown):      -{results['var_99_dd_pct']:.2f}%")
    print(f"  • 95% Conditional VaR (Expected Shortfall): -{results['cvar_95_dd_pct']:.2f}%")
    print(f"  • Probability of Drawdown > 5.0%:        {results['prob_dd_over_5pct']:.2f}%")
    print(f"  • Ruin Probability (Drawdown > 10.0%):   {results['prob_dd_over_10pct']:.2f}%")
    print("=" * 65)

    chart_path = ROOT_DIR / "reports" / "monte_carlo_stress_test.png"
    plot_stress_test_results(results, chart_path)


if __name__ == "__main__":
    main()
