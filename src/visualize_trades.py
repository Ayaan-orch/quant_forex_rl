import torch as th
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from environment import AdvancedForexEnv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "processed_market_features.csv"
MODEL_PATH = ROOT_DIR / "ppo_forex_policy"
OUTPUT_HTML = ROOT_DIR / "trade_executions.html"

def load_policy(model_path: Path):
    try:
        model = RecurrentPPO.load(str(model_path))
        is_recurrent = True
    except Exception:
        model = PPO.load(str(model_path))
        is_recurrent = False
    return model, is_recurrent

def generate_interactive_chart():
    print(f"\n[Trade Visualizer] Generating interactive Plotly execution map...")
    df = pd.read_csv(DATA_PATH)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    env = AdvancedForexEnv(test_df, random_start=False)
    model, is_recurrent = load_policy(MODEL_PATH)

    obs, _ = env.reset()
    terminated = False
    trade_logs = []

    if is_recurrent:
        state = (np.zeros(model.policy.lstm_hidden_state_shape), np.zeros(model.policy.lstm_hidden_state_shape))
    episode_starts = np.array([True])

    while not terminated:
        decision_idx = env.current_step
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

        if info["action"] == 1:
            trade_logs.append({
                "entry_bar": decision_idx,
                "exit_bar": info["step"],
                "direction": info["direction"],
                "return": info["step_return"],
                "friction": info.get("friction_paid", 0.0)
            })

    fig = go.Figure()

    # Base Candlestick Chart (sampled to last 500 bars for fast rendering or full if desired)
    fig.add_trace(go.Candlestick(
        x=test_df.index,
        open=test_df['Open'],
        high=test_df['High'],
        low=test_df['Low'],
        close=test_df['Close'],
        name='EUR/USD Market Price'
    ))

    # Plot Trade Overlays
    for t in trade_logs:
        color = '#00f0ff' if t['return'] > 0 else '#ff0055'
        marker_symbol = 'triangle-up' if t['direction'] == 1 else 'triangle-down'
        label = f"{'Long' if t['direction'] == 1 else 'Short'} ({t['return']*100:+.2f}%)"

        fig.add_trace(go.Scatter(
            x=[t['entry_bar']],
            y=[test_df.loc[t['entry_bar'], 'Close']],
            mode='markers',
            marker=dict(symbol=marker_symbol, size=14, color=color, line=dict(width=1.5, color="white")),
            name=label,
            hoverinfo='text',
            text=f"Action: {'BUY' if t['direction'] == 1 else 'SELL'}<br>Return: {t['return']*100:+.2f}%<br>Friction: {t['friction']*10000:.1f} pips",
            showlegend=False
        ))

    fig.update_layout(
        title="Out-of-Sample Trade Execution Map (Recurrent PPO Gatekeeper)",
        xaxis_title="Out-of-Sample Candle Bar Index",
        yaxis_title="EUR/USD Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    fig.write_html(str(OUTPUT_HTML))
    print(f"Interactive visual report saved to '{OUTPUT_HTML}'.")

if __name__ == "__main__":
    generate_interactive_chart()