import torch as th
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from environment import AdvancedForexEnv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "processed_market_features.csv"
MODEL_PATH = ROOT_DIR / "ppo_forex_policy"
PLOT_PATH = ROOT_DIR / "backtest_performance.png"

def load_policy(model_path: Path):
    """Loads either RecurrentPPO or standard PPO policy seamlessly."""
    try:
        model = RecurrentPPO.load(str(model_path))
        is_recurrent = True
        print(f"Loaded RecurrentPPO (LSTM) policy from {model_path}.")
    except Exception:
        model = PPO.load(str(model_path))
        is_recurrent = False
        print(f"Loaded Standard PPO policy from {model_path}.")
    return model, is_recurrent

def run_backtest():
    print(f"\n[Backtest Engine] Loading market data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 20% out-of-sample test partition
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"Evaluating Out-of-Sample performance on {len(test_df):,} bars (~{len(test_df)//24} trading days)...")
    test_env = AdvancedForexEnv(test_df, random_start=False)
    model, is_recurrent = load_policy(MODEL_PATH)

    obs, _ = test_env.reset()
    terminated = False
    
    trade_records = []
    actions_taken = []
    directions_traded = []
    frictions_paid = []

    # LSTM state tracking
    if is_recurrent:
        state = (np.zeros(model.policy.lstm_hidden_state_shape), np.zeros(model.policy.lstm_hidden_state_shape))
    episode_starts = np.array([True])

    while not terminated:
        current_bar_idx = test_env.current_step
        
        if is_recurrent:
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            with th.no_grad():
                states = (th.tensor(state[0], dtype=th.float32), th.tensor(state[1], dtype=th.float32))
                ep_tensor = th.tensor(episode_starts, dtype=th.float32)
                dist, new_states = model.policy.get_distribution(obs_tensor, states, ep_tensor)
                p_exec = float(dist.distribution.probs.cpu().numpy()[0][1])
                state = (new_states[0].cpu().numpy(), new_states[1].cpu().numpy())
            episode_starts = np.array([False])
            # Institutional Conviction Gate: Execute only when model conviction >= 0.88
            action = 1 if p_exec >= 0.88 else 0
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = test_env.step(int(action))
        
        actions_taken.append(int(action))
        frictions_paid.append(info.get("friction_paid", 0.0))

        if info["action"] == 1:
            trade_records.append({
                "entry_bar": current_bar_idx,
                "exit_bar": info["step"],
                "direction": info["direction"],
                "return": info["step_return"],
                "friction": info.get("friction_paid", 0.0)
            })
            directions_traded.append(info["direction"])
        else:
            trade_records.append({
                "entry_bar": current_bar_idx,
                "exit_bar": info["step"],
                "direction": 0,
                "return": 0.0,
                "friction": 0.0
            })

    returns_arr = np.array([t["return"] for t in trade_records])
    active_trades = np.array([t["return"] for t in trade_records if t["direction"] != 0])
    
    # Portfolio equity calculation
    equity_curve = np.cumprod(1.0 + returns_arr)
    cumulative_strategy = equity_curve - 1.0
    
    # Chronological asset benchmark return
    if 'symbol' in test_df.columns:
        sym_rets = []
        for s in test_df['symbol'].unique():
            s_df = test_df[test_df['symbol'] == s]
            if len(s_df) > 1:
                sym_rets.append((s_df['Close'].iloc[-1] - s_df['Close'].iloc[0]) / s_df['Close'].iloc[0])
        underlying_return = np.mean(sym_rets) if sym_rets else 0.0
    else:
        underlying_return = (test_df["Close"].iloc[-1] - test_df["Close"].iloc[0]) / test_df["Close"].iloc[0]

    # Win / Loss Metrics
    win_trades = active_trades[active_trades > 0]
    loss_trades = active_trades[active_trades < 0]
    win_count = len(win_trades)
    loss_count = len(loss_trades)
    total_active = len(active_trades)
    win_rate = (win_count / total_active * 100) if total_active > 0 else 0.0

    gross_gains = np.sum(win_trades) if win_count > 0 else 0.0
    gross_losses = abs(np.sum(loss_trades)) if loss_count > 0 else 1e-9
    profit_factor = gross_gains / gross_losses

    # Drawdown Metrics
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_drawdown = np.min(drawdowns) * 100 if len(drawdowns) > 0 else 0.0

    # Risk-Adjusted Return Metrics (Annualized on H4 horizon: 6 bars/day * 252 days = 1,512 bars/year)
    annual_factor = np.sqrt(1512 / max(1, len(trade_records)))
    mean_ret = np.mean(active_trades) if total_active > 0 else 0.0
    std_ret = np.std(active_trades) if total_active > 1 else 1.0
    
    trade_sharpe = (mean_ret / (std_ret + 1e-9)) * annual_factor if std_ret > 0 else 0.0
    downside_std = np.std(loss_trades) if len(loss_trades) > 1 else 1e-9
    sortino_ratio = (mean_ret / downside_std) * annual_factor if downside_std > 0 else 0.0
    calmar_ratio = abs(cumulative_strategy[-1] / (max_drawdown / 100.0 + 1e-9)) if max_drawdown < 0 else 0.0

    pass_pct = np.mean(np.array(actions_taken) == 0) * 100
    take_pct = np.mean(np.array(actions_taken) == 1) * 100
    long_count = sum(1 for d in directions_traded if d == 1)
    short_count = sum(1 for d in directions_traded if d == -1)

    print("\n" + "=" * 58)
    print("         INSTITUTIONAL GATEKEEPER PERFORMANCE REPORT")
    print("=" * 58)
    print(f"Test Partition Horizon      : {len(test_df):,} H4 bars (~{len(test_df)//6} trading days)")
    print(f"Candidate Setups Screened   : {len(actions_taken):,}")
    print(f"Trades Executed             : {total_active} (Longs: {long_count}, Shorts: {short_count})")
    print(f"Setup Gatekeeping Rate      : Pass {pass_pct:.1f}% | Execute {take_pct:.1f}%")
    print("-" * 58)
    print(f"Win Rate (Executed Trades)  : {win_rate:.1f}% ({win_count}W / {loss_count}L)")
    print(f"Profit Factor               : {profit_factor:.2f}")
    print(f"Max Strategy Drawdown       : {max_drawdown:.2f}%")
    print(f"Cumulative Net Return       : {cumulative_strategy[-1] * 100:+.2f}%")
    print(f"Underlying Buy & Hold Return: {underlying_return * 100:+.2f}%")
    print(f"Annualized Sharpe Ratio     : {trade_sharpe:.2f}")
    print(f"Sortino Ratio               : {sortino_ratio:.2f}")
    print(f"Calmar Ratio                : {calmar_ratio:.2f}")
    total_pips = sum(
        (t['friction'] / test_df.iloc[t['entry_bar']]['Close']) * (100.0 if test_df.iloc[t['entry_bar']]['Close'] > 50 else 10000.0)
        for t in trade_records if t['direction'] != 0
    )
    print(f"Total Friction Paid (Pips)  : {total_pips:.1f} pips")
    print("=" * 58)

    # Plot Multi-Panel Institutional Diagnostics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    
    # 1. Equity Curve
    ax1.plot(cumulative_strategy * 100, label="PPO Gatekeeper Strategy Net Return (%)", color="#00a86b", lw=2)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Out-of-Sample Performance & Drawdown Profile", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Net Return (%)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Underwater Drawdown Curve
    ax2.fill_between(range(len(drawdowns)), drawdowns * 100, 0, color="#d90429", alpha=0.4)
    ax2.plot(drawdowns * 100, color="#d90429", lw=1.2)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Trade / Decision Sequence Index")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(str(PLOT_PATH), dpi=220)
    print(f"\nDiagnostic performance plot saved to '{PLOT_PATH}'.")

if __name__ == "__main__":
    run_backtest()