import os
import time
import ctypes
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import requests
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
LOG_FILE = DATA_DIR / "live_trades_log.csv"
DB_FILE = DATA_DIR / "trades_telemetry.db"
MODEL_PATH = ROOT_DIR / "ppo_forex_policy"

# Load environment variables
env_path = ROOT_DIR / ".env"
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
MAGIC_NUMBER = 101088
MAX_DAILY_DRAWDOWN_PCT = 0.03  # 3.0% Max Daily Loss Circuit Breaker

# Windows sleep management flags
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def prevent_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("Power management: System sleep lock enabled.")
    except Exception as e:
        print(f"Could not configure sleep prevention: {e}")

def restore_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass

def send_discord_alert(title: str, description: str, color: int = 3447003):
    if not DISCORD_WEBHOOK_URL:
        return
    payload = {
        "embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }]
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        print(f"\n[Discord Warning] Could not dispatch notification: {e}")

def init_telemetry_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                event_type TEXT,
                price REAL,
                trend_pct REAL,
                norm_rsi REAL,
                norm_atr_pct REAL,
                decision TEXT,
                direction INTEGER,
                order_ticket TEXT,
                status TEXT
            )
        """)
        conn.commit()

def log_event(symbol, event_type, price, trend, rsi, atr, decision, direction, ticket=None, status="OK"):
    init_telemetry_db()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. SQLite Persistent Thread-Safe Log
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO telemetry 
                (timestamp, symbol, event_type, price, trend_pct, norm_rsi, norm_atr_pct, decision, direction, order_ticket, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ts, symbol, event_type, float(price), float(trend) * 100, float(rsi), float(atr) * 100, decision, direction, str(ticket) if ticket else "-", status))
            conn.commit()
    except Exception as e:
        print(f"DB Error: {e}")

    # 2. CSV Backward Compatibility
    record = {
        "timestamp": ts, "symbol": symbol, "event_type": event_type,
        "price": round(float(price), 5), "trend_pct": round(float(trend) * 100, 4),
        "norm_rsi": round(float(rsi), 4), "norm_atr_pct": round(float(atr) * 100, 4),
        "decision": decision, "direction": direction,
        "order_ticket": ticket if ticket else "-", "status": status
    }
    pd.DataFrame([record]).to_csv(LOG_FILE, mode='a', header=not LOG_FILE.exists(), index=False)

def initialize_mt5():
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    
    first_attempt = True
    while True:
        if login and password and server:
            connected = mt5.initialize(login=int(login), password=password, server=server)
        else:
            connected = mt5.initialize()
            
        if connected:
            break
            
        err = mt5.last_error()
        if first_attempt:
            print(f"\nMT5 Connection failed: {err}")
            if err[0] == -6:
                print(">>> [MT5 Action Required]: Error -6 means MetaTrader 5 is open but NOT logged into an active broker account.")
                print(">>> Please switch to your open MetaTrader 5 window and log into or open a free Demo account:")
                print(">>>   File -> 'Open an Account' -> 'MetaQuotes-Demo' -> 'Open a demo account' -> Finish.")
                print(">>> Ensure the bottom-right corner of MT5 shows green connection speed bars.\n")
            first_attempt = False
        else:
            print(f"Waiting for MT5 account login... (Last error: {err[1]}). Retrying in 10s...")
            
        time.sleep(10)
        
    print("Connected to MetaTrader 5 Terminal successfully.")
    mt5.symbol_select(SYMBOL, True)

def get_filling_mode(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_FOK
    if info.filling_mode & 1:
        return mt5.ORDER_FILLING_FOK
    elif info.filling_mode & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN

def fetch_live_features():
    """
    Fetches historical rates and computes technical, causal macro, and Daily HTF features
    with exact parity to the training environment.
    """
    try:
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 150)
        if rates is None or len(rates) < 60:
            return None, None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)

        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))

        # Wilder's RSI (14)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.ewm(alpha=1.0/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        raw_rsi = 100.0 - (100.0 / (1.0 + rs))
        df['norm_rsi'] = (raw_rsi - 50.0) / 50.0

        # Wilder's ATR (14)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        raw_atr = tr.ewm(alpha=1.0/14, min_periods=14, adjust=False).mean()
        df['norm_atr'] = raw_atr / df['Close']

        # Normalized Activity / Volume Z-Score
        vol_mean = df['Volume'].rolling(24).mean()
        vol_std = df['Volume'].rolling(24).std()
        df['norm_vol'] = (df['Volume'] - vol_mean) / (vol_std + 1e-9)

        # EMA Trend Spread
        ema_20 = df['Close'].ewm(span=20, adjust=False).mean()
        ema_50 = df['Close'].ewm(span=50, adjust=False).mean()
        df['trend_spread'] = (ema_20 - ema_50) / df['Close']

        # Causal shock & time proximity parity
        hourly_range = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
        rolling_p90 = hourly_range.rolling(window=min(len(df), 168), min_periods=24).quantile(0.90).bfill()
        vol_shock = hourly_range >= rolling_p90
        
        last_shock = -9999
        time_prox = np.ones(len(df), dtype=np.float32)
        for i in range(len(df)):
            if vol_shock.iloc[i]:
                last_shock = i
            bars_since = i - last_shock
            time_prox[i] = min(bars_since / 24.0, 1.0)
            
        df['time_to_event'] = time_prox
        df['event_surprise'] = 0.0
        shock_indices = np.where(vol_shock.shift(1).fillna(False))[0]
        for idx in shock_indices:
            ratio = hourly_range.iloc[idx - 1] / (rolling_p90.iloc[idx - 1] + 1e-9)
            df.loc[idx, 'event_surprise'] = float(np.clip(np.sign(df['log_ret'].iloc[idx]) * ratio, -3.0, 3.0))

        rsi_sentiment = df['norm_rsi'] * 0.5
        mom_sentiment = np.tanh(df['log_ret'].rolling(12, min_periods=1).mean() * 100) * 0.5
        df['sentiment_score'] = (rsi_sentiment + mom_sentiment).fillna(0.0)

        # Higher Timeframe Daily EMA 200 & RSI
        rates_d1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 1, 205)
        if rates_d1 is not None and len(rates_d1) >= 200:
            df_d1 = pd.DataFrame(rates_d1)
            ema200 = df_d1['close'].ewm(span=200, adjust=False).mean().iloc[-1]
            last_d1_close = df_d1['close'].iloc[-1]
            df['htf_trend_200'] = (last_d1_close - ema200) / last_d1_close
            # D1 RSI
            d1_delta = df_d1['close'].diff()
            d1_gain = d1_delta.where(d1_delta > 0, 0.0).ewm(alpha=1.0/14, min_periods=14, adjust=False).mean()
            d1_loss = (-d1_delta.where(d1_delta < 0, 0.0)).ewm(alpha=1.0/14, min_periods=14, adjust=False).mean()
            d1_rs = d1_gain.iloc[-1] / (d1_loss.iloc[-1] + 1e-9)
            df['htf_rsi'] = ((100.0 - (100.0 / (1.0 + d1_rs))) - 50.0) / 50.0
        else:
            df['htf_trend_200'] = 0.0
            df['htf_rsi'] = 0.0

        return df.iloc[-1], df.iloc[-2]
    except Exception as e:
        print(f"\nError computing features: {e}")
        return None, None

def calculate_position_size(symbol: str, sl_distance_price: float, risk_pct: float = 0.01) -> float:
    """
    Calculates institutional volatility-adjusted position size.
    Risks exactly risk_pct (e.g. 1.0%) of account equity based on the SL distance.
    """
    account_info = mt5.account_info()
    if account_info is None:
        return 0.01
    equity = account_info.equity
    risk_amount = equity * risk_pct

    sym_info = mt5.symbol_info(symbol)
    if sym_info is None or sym_info.point <= 0:
        return 0.01

    point = sym_info.point
    tick_val = sym_info.trade_tick_value if sym_info.trade_tick_value > 0 else 1.0
    sl_points = sl_distance_price / point
    loss_per_lot = sl_points * tick_val

    if loss_per_lot <= 0:
        return 0.01

    raw_lot = risk_amount / loss_per_lot
    min_lot = sym_info.volume_min
    max_lot = sym_info.volume_max
    step_lot = sym_info.volume_step if sym_info.volume_step > 0 else 0.01

    lot = max(min_lot, min(max_lot, round(raw_lot / step_lot) * step_lot))
    return round(lot, 2)

def send_market_order(direction, raw_atr, current_bar):
    sl_dist = 1.0 * raw_atr
    lot_size = calculate_position_size(SYMBOL, sl_dist, risk_pct=0.01)
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return

    if direction == 1:
        order_type = mt5.ORDER_TYPE_BUY
        entry_price = tick.ask
        tp = entry_price + (1.6 * raw_atr)
        sl = entry_price - (1.0 * raw_atr)
    else:
        order_type = mt5.ORDER_TYPE_SELL
        entry_price = tick.bid
        tp = entry_price - (1.6 * raw_atr)
        sl = entry_price + (1.0 * raw_atr)

    filling_type = get_filling_mode(SYMBOL)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot_size,
        "type": order_type,
        "price": entry_price,
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "PPO_Forex_RL",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        err_msg = f"Order failed: {result.retcode} ({result.comment})"
        print(f"\n{err_msg}")
        log_event(SYMBOL, "ORDER_ERROR", entry_price, current_bar['trend_spread'], current_bar['norm_rsi'], current_bar['norm_atr'], "EXECUTE_FAILED", direction, status=err_msg)
        send_discord_alert("❌ Execution Error", f"Failed to execute {SYMBOL} order:\n`{err_msg}`", color=15548997)
    else:
        print(f"\nORDER PLACED! Direction: {'BUY' if direction == 1 else 'SELL'} | Ticket: #{result.order} | Entry: {entry_price:.5f} | TP: {tp:.5f} | SL: {sl:.5f}")
        log_event(SYMBOL, "TRADE_EXECUTED", entry_price, current_bar['trend_spread'], current_bar['norm_rsi'], current_bar['norm_atr'], "CONFIRMED", direction, ticket=result.order, status="FILLED")
        
        action_name = "BUY (Long)" if direction == 1 else "SELL (Short)"
        alert_body = (
            f"**Symbol:** `{SYMBOL}`\n"
            f"**Direction:** `{action_name}`\n"
            f"**Ticket:** `#{result.order}`\n"
            f"**Entry Price:** `{entry_price:.5f}`\n"
            f"**Take Profit:** `{tp:.5f}`\n"
            f"**Stop Loss:** `{sl:.5f}`\n"
            f"**Target Bracket:** `+1.6R / -1.0R` (`~{raw_atr*16000:.1f}` pips TP)"
        )
        send_discord_alert(f"🚀 Trade Executed: {action_name}", alert_body, color=5763719)

def run_live_daemon():
    print("=" * 60)
    print(f"  QUANT FOREX RL — LIVE PRODUCTION DAEMON ({SYMBOL})")
    print(f"  Circuit Breaker: {MAX_DAILY_DRAWDOWN_PCT*100:.1f}% Max Daily Loss | Magic: #{MAGIC_NUMBER}")
    print("=" * 60)

    prevent_sleep()
    initialize_mt5()
    init_telemetry_db()

    # Load Model (RecurrentPPO with fallback)
    try:
        model = RecurrentPPO.load(str(MODEL_PATH))
        is_recurrent = True
        print(f"Loaded RecurrentPPO (LSTM) policy from {MODEL_PATH}.")
    except Exception:
        model = PPO.load(str(MODEL_PATH))
        is_recurrent = False
        print(f"Loaded standard PPO policy from {MODEL_PATH}.")

    # State Recovery Check
    open_positions = mt5.positions_get(symbol=SYMBOL)
    if open_positions:
        print(f"\n[State Recovery] Detected {len(open_positions)} active position(s) in MT5. Managing existing trades.")
        for p in open_positions:
            print(f" -> Ticket #{p.ticket}: {'BUY' if p.type == 0 else 'SELL'} {p.volume} lots @ {p.price_open:.5f} (Profit: ${p.profit:+.2f})")
    else:
        print("\nNo existing positions open. Ready for candidate screening.")

    account_info = mt5.account_info()
    starting_balance = account_info.balance if account_info else 10000.0
    current_day = datetime.now().date()

    last_processed_candle_time = None
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    try:
        while True:
            # Daily Circuit Breaker Refresh
            now_day = datetime.now().date()
            if now_day != current_day:
                current_day = now_day
                acc = mt5.account_info()
                if acc: starting_balance = acc.balance
                print(f"\n[Daily Rollover] Refreshed daily starting balance: ${starting_balance:,.2f}")

            # Check Circuit Breaker
            acc = mt5.account_info()
            if acc:
                daily_dd = (acc.equity - starting_balance) / starting_balance
                if daily_dd <= -MAX_DAILY_DRAWDOWN_PCT:
                    print(f"\n🛑 CIRCUIT BREAKER TRIGGERED: Daily Drawdown at {daily_dd*100:.2f}% (Limit: -{MAX_DAILY_DRAWDOWN_PCT*100:.1f}%). Halting orders.")
                    time.sleep(60)
                    continue

            forming_bar, closed_bar = fetch_live_features()
            if forming_bar is not None and closed_bar is not None:
                candle_time = forming_bar['time']
                trend = forming_bar['trend_spread']
                rsi = forming_bar['norm_rsi']
                atr = forming_bar['norm_atr']
                price = forming_bar['Close']

                print(
                    f"\r[{datetime.now().strftime('%H:%M:%S')}] {SYMBOL}: {price:.5f} | "
                    f"Trend: {trend*10000:+.1f} pips | RSI: {rsi:+.2f} | ATR: {atr*10000:.1f} pips | Eq: ${acc.equity:,.2f}",
                    end="",
                    flush=True
                )

                if candle_time != last_processed_candle_time:
                    last_processed_candle_time = candle_time
                    print(f"\nEvaluating closed candle {closed_bar['time']} for {SYMBOL}...")

                    c_atr = closed_bar['norm_atr']
                    c_trend = closed_bar['trend_spread']
                    c_rsi = closed_bar['norm_rsi']
                    c_price = closed_bar['Close']
                    c_htf = closed_bar.get('htf_trend_200', 0.0)

                    candidate_dir = 0
                    if c_atr >= 0.00075:
                        if c_trend > 0.00015 and c_rsi < 0.10 and c_htf >= -0.005:
                            candidate_dir = 1
                        elif c_trend < -0.00015 and c_rsi > -0.10 and c_htf <= 0.005:
                            candidate_dir = -1

                    if candidate_dir != 0:
                        obs = np.array([
                            closed_bar['log_ret'],
                            closed_bar['norm_rsi'],
                            closed_bar['norm_atr'],
                            closed_bar['norm_vol'],
                            closed_bar['trend_spread'],
                            closed_bar['sentiment_score'],
                            closed_bar['event_surprise'],
                            closed_bar['time_to_event'],
                            closed_bar['htf_trend_200'],
                            closed_bar['htf_rsi'],
                            float(candidate_dir)
                        ], dtype=np.float32)

                        if is_recurrent:
                            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                            episode_starts = np.zeros((1,), dtype=bool)
                        else:
                            action, _ = model.predict(obs, deterministic=True)

                        if action == 1:
                            positions = mt5.positions_get(symbol=SYMBOL)
                            if positions is None or len(positions) == 0:
                                raw_atr = c_atr * c_price
                                send_market_order(candidate_dir, raw_atr, closed_bar)
                            else:
                                print(" -> Setup confirmed by agent, but position already open. Holding.")
                                log_event(SYMBOL, "SETUP_SKIPPED_DUPLICATE", c_price, c_trend, c_rsi, c_atr, "CONFIRMED_HELD", candidate_dir)
                        else:
                            print(" -> Setup rejected by PPO Gatekeeper (Action: Pass).")
                            log_event(SYMBOL, "SETUP_FILTERED", c_price, c_trend, c_rsi, c_atr, "REJECTED_PASS", candidate_dir)
                    else:
                        print(" -> No candidate setup on closed H1 candle. Monitoring...")
                        log_event(SYMBOL, "REGIME_CHECK", c_price, c_trend, c_rsi, c_atr, "NO_SETUP", 0, status="MONITORING")

            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping live execution daemon...")
        send_discord_alert("🔴 Trading Daemon Stopped", "Daemon received manual interrupt signal and shut down cleanly.", color=15548997)
    finally:
        restore_sleep()
        mt5.shutdown()
        print("Clean shutdown complete.")

if __name__ == "__main__":
    run_live_daemon()