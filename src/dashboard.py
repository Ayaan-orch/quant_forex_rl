import os
import sqlite3
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
LOG_FILE = DATA_DIR / "live_trades_log.csv"
DB_FILE = DATA_DIR / "trades_telemetry.db"
PLOT_FILE = ROOT_DIR / "backtest_performance.png"
STRESS_PLOT_FILE = ROOT_DIR / "reports" / "monte_carlo_stress_test.png"

st.set_page_config(
    page_title="Forex RL Telemetry & Risk Monitor", 
    page_icon="⚡", 
    layout="wide"
)

st.title("⚡ Quant Forex RL — Live Telemetry & Risk Monitor")

# Top Metrics Row (Live MT5 Connection)
col1, col2, col3, col4, col5 = st.columns(5)

mt5_connected = mt5.initialize() if mt5 is not None else False
if mt5_connected:
    account_info = mt5.account_info()
    positions = mt5.positions_get()
    
    if account_info:
        col1.metric("Account Balance", f"${account_info.balance:,.2f}")
        col2.metric("Floating Equity", f"${account_info.equity:,.2f}")
        col3.metric("Floating PnL", f"${account_info.profit:+,.2f}")
        col4.metric("Active Positions", len(positions) if positions else 0)
        
        # Margin Level
        margin_lvl = account_info.margin_level if account_info.margin_level else 0.0
        col5.metric("Margin Level", f"{margin_lvl:,.1f}%" if margin_lvl > 0 else "N/A")

    # Live Exposure Table
    st.subheader("📊 Live Open MT5 Positions")
    if positions and len(positions) > 0:
        pos_data = [{
            "Ticket": p.ticket,
            "Symbol": p.symbol,
            "Type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
            "Volume": p.volume,
            "Open Price": p.price_open,
            "Current Price": p.price_current,
            "Stop Loss": p.sl,
            "Take Profit": p.tp,
            "Profit ($)": round(p.profit, 2)
        } for p in positions]
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
    else:
        st.info("No active open positions in MT5 terminal.")
    mt5.shutdown()
else:
    col1.metric("MT5 Status", "Offline")
    st.caption("MetaTrader 5 terminal offline or not running locally. Displaying persistent SQLite telemetry.")

st.divider()

# Telemetry History Section (SQLite with CSV Fallback)
st.subheader("📜 Strategy Telemetry & Candidate Decisions")

telemetry_df = None
if DB_FILE.exists():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            telemetry_df = pd.read_sql_query("SELECT * FROM telemetry ORDER BY id DESC LIMIT 100", conn)
    except Exception as e:
        st.warning(f"Error reading SQLite database: {e}")

if telemetry_df is None or telemetry_df.empty:
    if LOG_FILE.exists():
        try:
            telemetry_df = pd.read_csv(LOG_FILE)
        except Exception:
            pass

if telemetry_df is not None and not telemetry_df.empty:
    m1, m2, m3, m4 = st.columns(4)
    total_events = len(telemetry_df)
    trades_executed = len(telemetry_df[telemetry_df["event_type"] == "TRADE_EXECUTED"])
    filtered_count = len(telemetry_df[telemetry_df["decision"] == "REJECTED_PASS"])
    pass_rate = (filtered_count / max(1, trades_executed + filtered_count)) * 100

    m1.metric("Screened Events Logged", total_events)
    m2.metric("Executed Trades", trades_executed)
    m3.metric("Filtered Setups (Passed)", filtered_count)
    m4.metric("Gatekeeper Pass Rate", f"{pass_rate:.1f}%")

    st.dataframe(telemetry_df.head(25), use_container_width=True)

    # Volatility Chart
    if "norm_atr_pct" in telemetry_df.columns:
        fig_atr = px.line(
            telemetry_df,
            x="timestamp",
            y="norm_atr_pct",
            color="symbol",
            title="Real-Time Volatility Regime (ATR %)"
        )
        fig_atr.add_hline(y=0.075, line_dash="dash", line_color="red", annotation_text="0.075% Minimum Setup Volatility Threshold")
        st.plotly_chart(fig_atr, use_container_width=True)
else:
    st.info("No telemetry logs found yet. Start `src/live_execution.py` to begin recording live telemetry.")

st.divider()

# Backtest & Stress Test Performance Artifacts
st.subheader("📈 Institutional Diagnostics & Stress-Testing")

tab_backtest, tab_monte_carlo, tab_risk = st.tabs([
    "📈 Out-of-Sample Performance", 
    "🎲 Monte Carlo Stress Test & VaR", 
    "🛡️ Risk Management Specifications"
])

with tab_backtest:
    if PLOT_FILE.exists():
        st.image(str(PLOT_FILE), caption="Out-of-Sample Performance Curve & Drawdown Profile", use_container_width=True)
    else:
        st.info("Run `src/backtest.py` to generate the latest out-of-sample performance chart.")

with tab_monte_carlo:
    if STRESS_PLOT_FILE.exists():
        st.image(str(STRESS_PLOT_FILE), caption="2,500 Monte Carlo Bootstrap Trajectories & Value at Risk (VaR)", use_container_width=True)
    else:
        st.info("Run `src/stress_test.py` to generate the latest Monte Carlo stress-testing distribution.")

with tab_risk:
    st.markdown("""
    ### 🛡️ Production Risk Controls & Edge Specifications
    * **Policy Network**: Recurrent PPO (`sb3_contrib.RecurrentPPO` / `MlpLstmPolicy`) with 64-unit temporal LSTM
    * **Conviction Gating**: Orders execute strictly when softmax probability $P(\text{Trade}) \ge 0.88$ (rejects 84% chop)
    * **Asymmetric Execution**: 2.5 : 1 Reward-to-Risk bracket with dynamic Breakeven (+1.0 ATR) and Trailing Stop (+1.6 ATR)
    * **Currency Correlation Guard**: Net directional USD exposure capped at 1 position across portfolio
    * **High-Impact News Shield**: Automatic order lockout during US NFP and FOMC rate decision windows
    * **Weekend Protection**: Auto-lockout on Friday after 19:00 UTC through Sunday pre-market
    * **Position Sizing**: Volatility-adjusted 1.0% Fixed Fractional Equity Risk
    * **Drawdown Circuit Breaker**: `-3.0%` Daily Floating Drawdown Emergency Halt
    * **Telemetry Database**: Persistent, thread-safe SQLite telemetry (`data/trades_telemetry.db`)
    """)

if st.button("🔄 Refresh Telemetry Dashboard"):
    st.rerun()