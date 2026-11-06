import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import MetaTrader5 as mt5
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_PATH = DATA_DIR / "processed_market_features.csv"

# Target Multi-Asset Universe for High Trend & Liquidity
SYMBOLS_CONFIG = [
    {"mt5": "USDJPY", "yf": "JPY=X", "name": "USDJPY"},
    {"mt5": "EURUSD", "yf": "EURUSD=X", "name": "EURUSD"},
    {"mt5": "GBPUSD", "yf": "GBPUSD=X", "name": "GBPUSD"},
]

def fetch_data_source(symbol_cfg: dict, period: str = "2y") -> pd.DataFrame:
    """Fetches H1 historical data from MT5 if connected, else via Yahoo Finance."""
    mt5_sym = symbol_cfg["mt5"]
    yf_sym = symbol_cfg["yf"]

    # 1. Try MT5
    if mt5.initialize():
        try:
            mt5.symbol_select(mt5_sym, True)
            rates = mt5.copy_rates_from_pos(mt5_sym, mt5.TIMEFRAME_H1, 0, 12500)
            if rates is not None and len(rates) > 1000:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low',
                    'close': 'Close', 'tick_volume': 'Volume'
                }, inplace=True)
                print(f"[MT5] Retrieved {len(df):,} H1 bars for {mt5_sym}.")
                return df[['time', 'Open', 'High', 'Low', 'Close', 'Volume']]
        finally:
            mt5.shutdown()

    # 2. Fallback to yfinance
    print(f"[Fallback] Downloading {yf_sym} ({symbol_cfg['name']}) via Yahoo Finance...")
    df_yf = yf.download(yf_sym, period=period, interval="1h", progress=False)
    if df_yf is not None and not df_yf.empty:
        # Flatten MultiIndex columns if present
        if isinstance(df_yf.columns, pd.MultiIndex):
            df_yf.columns = [col[0] for col in df_yf.columns]
        df_yf.reset_index(inplace=True)
        # Rename Datetime / Date column to time
        date_col = 'Datetime' if 'Datetime' in df_yf.columns else ('Date' if 'Date' in df_yf.columns else df_yf.columns[0])
        df_yf.rename(columns={date_col: 'time'}, inplace=True)
        df_yf['time'] = pd.to_datetime(df_yf['time']).dt.tz_localize(None)
        print(f"[yfinance] Retrieved {len(df_yf):,} H1 bars for {symbol_cfg['name']}.")
        return df_yf[['time', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    raise RuntimeError(f"Could not retrieve historical data for {symbol_cfg['name']}.")

def resample_to_h4(df: pd.DataFrame) -> pd.DataFrame:
    """Resamples H1 OHLCV bars to H4 (4-Hour) execution bars."""
    data = df.copy()
    data['time'] = pd.to_datetime(data['time'])
    data.set_index('time', inplace=True)
    
    h4 = pd.DataFrame()
    h4['Open'] = data['Open'].resample('4h').first()
    h4['High'] = data['High'].resample('4h').max()
    h4['Low'] = data['Low'].resample('4h').min()
    h4['Close'] = data['Close'].resample('4h').last()
    h4['Volume'] = data['Volume'].resample('4h').sum()
    h4.dropna(inplace=True)
    h4.reset_index(inplace=True)
    return h4

def compute_h4_features(df_h4: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    """Computes normalized stationary indicators and Daily macro trend on H4 bars."""
    data = df_h4.copy()
    data['symbol'] = asset_name

    # 1. Log Return
    data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1)).fillna(0.0)

    # 2. Normalized RSI (14) centered at 0 in [-1.0, 1.0]
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1.0/14, min_periods=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1.0/14, min_periods=14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    raw_rsi = 100.0 - (100.0 / (1.0 + rs))
    data['norm_rsi'] = ((raw_rsi - 50.0) / 50.0).fillna(0.0)

    # 3. ATR (14) & Normalized ATR
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift(1)).abs()
    low_close = (data['Low'] - data['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['raw_atr'] = tr.ewm(alpha=1.0/14, min_periods=14, adjust=False).mean()
    data['norm_atr'] = (data['raw_atr'] / data['Close']).fillna(0.001)

    # 4. Volume Z-Score (18 bars = 3 trading days on H4)
    vol_mean = data['Volume'].rolling(18, min_periods=6).mean()
    vol_std = data['Volume'].rolling(18, min_periods=6).std()
    data['norm_vol'] = ((data['Volume'] - vol_mean) / (vol_std + 1e-9)).fillna(0.0)

    # 5. EMA 20/50 Trend Spread
    ema_20 = data['Close'].ewm(span=20, adjust=False).mean()
    ema_50 = data['Close'].ewm(span=50, adjust=False).mean()
    data['trend_spread'] = ((ema_20 - ema_50) / data['Close']).fillna(0.0)

    # 6. Higher Timeframe Daily 200 EMA and Daily RSI (Shifted by 1 day - Zero Lookahead)
    data['dt'] = pd.to_datetime(data['time'])
    daily = data.set_index('dt')['Close'].resample('1D').last().dropna().to_frame()
    daily['d_ema200'] = daily['Close'].ewm(span=200, adjust=False).mean()
    daily['htf_trend_200_raw'] = (daily['Close'] - daily['d_ema200']) / daily['Close']

    d_delta = daily['Close'].diff()
    d_gain = d_delta.where(d_delta > 0, 0.0).ewm(alpha=1.0/14, min_periods=14, adjust=False).mean()
    d_loss = (-d_delta.where(d_delta < 0, 0.0)).ewm(alpha=1.0/14, min_periods=14, adjust=False).mean()
    d_rs = d_gain / (d_loss + 1e-9)
    daily['htf_rsi_raw'] = ((100.0 - (100.0 / (1.0 + d_rs))) - 50.0) / 50.0

    daily['date'] = daily.index.date
    daily['htf_trend_200'] = daily['htf_trend_200_raw'].shift(1)
    daily['htf_rsi'] = daily['htf_rsi_raw'].shift(1)

    data['date'] = data['dt'].dt.date
    merged = pd.merge(data, daily[['date', 'htf_trend_200', 'htf_rsi']], on='date', how='left')
    merged['htf_trend_200'] = merged['htf_trend_200'].bfill().fillna(0.0)
    merged['htf_rsi'] = merged['htf_rsi'].bfill().fillna(0.0)

    # 7. Session Factor (London/NY overlap highest liquidity)
    hours = merged['dt'].dt.hour
    session_factor = np.where((hours >= 12) & (hours < 16), 1.4,
                     np.where((hours >= 7) & (hours < 12), 1.2,
                     np.where((hours >= 16) & (hours < 21), 1.1,
                     np.where((hours >= 0) & (hours < 7), 0.85, 0.7))))
    merged['session_factor'] = session_factor

    # 8. Microstructure & Event Placeholders
    merged['sentiment_score'] = merged['norm_rsi'] * 0.5
    merged['event_surprise'] = 0.0
    merged['time_to_event'] = 1.0

    # Clean temporary columns
    merged.drop(columns=['dt', 'date'], inplace=True, errors='ignore')
    merged.dropna(inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged

def run_pipeline():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print("  QUANT FOREX RL — MULTI-ASSET H4 DATA PIPELINE")
    print("=" * 60)

    all_assets_df = []
    for cfg in SYMBOLS_CONFIG:
        name = cfg["name"]
        print(f"\n[Processing] Ingesting {name}...")
        try:
            df_raw = fetch_data_source(cfg, period="2y")
            df_h4 = resample_to_h4(df_raw)
            features_df = compute_h4_features(df_h4, name)
            print(f"  -> Generated {len(features_df):,} H4 feature bars for {name}.")
            all_assets_df.append(features_df)
        except Exception as e:
            print(f"  [Error] Failed to ingest {name}: {e}")

    if not all_assets_df:
        print("[Fatal] No assets ingested successfully.")
        return

    # Combine all assets chronologically
    combined = pd.concat(all_assets_df, ignore_index=True)
    # Sort chronologically by time
    combined['time_dt'] = pd.to_datetime(combined['time'])
    combined.sort_values(by=['time_dt', 'symbol'], inplace=True)
    combined.drop(columns=['time_dt'], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_csv(PROCESSED_DATA_PATH, index=False)
    print("\n" + "=" * 60)
    print(f"SUCCESS: Multi-Asset H4 dataset serialized to '{PROCESSED_DATA_PATH}'.")
    print(f"Total Combined Ready Bars: {len(combined):,} across {combined['symbol'].nunique()} assets.")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
