import time
import MetaTrader5 as mt5

SYMBOL = "EURUSD"

if not mt5.initialize():
    print(f"Failed to initialize MT5: {mt5.last_error()}")
    quit()

mt5.symbol_select(SYMBOL, True)
symbol_info = mt5.symbol_info(SYMBOL)

if symbol_info is None:
    print(f"Symbol {SYMBOL} not found.")
    mt5.shutdown()
    quit()

# Automatically resolve the broker's accepted filling mode
filling_type = mt5.ORDER_FILLING_FOK
if symbol_info.filling_mode & 1:  # FOK supported
    filling_type = mt5.ORDER_FILLING_FOK
elif symbol_info.filling_mode & 2:  # IOC supported
    filling_type = mt5.ORDER_FILLING_IOC
else:
    filling_type = mt5.ORDER_FILLING_RETURN

price = mt5.symbol_info_tick(SYMBOL).ask
point = symbol_info.point
lot = 0.01

print(f"Testing market order for {SYMBOL} at {price} with filling mode: {filling_type}...")

# 1. Send test buy order
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": SYMBOL,
    "volume": lot,
    "type": mt5.ORDER_TYPE_BUY,
    "price": price,
    "sl": price - 100 * point,
    "tp": price + 100 * point,
    "deviation": 20,
    "magic": 999999,
    "comment": "Test_Ping",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": filling_type,
}

result = mt5.order_send(request)

if result.retcode != mt5.TRADE_RETCODE_DONE:
    print(f"Order failed! Error code: {result.retcode} | {result.comment}")
else:
    print(f"SUCCESS: Test position #{result.order} opened cleanly!")
    time.sleep(2)

    # 2. Close test position
    close_price = mt5.symbol_info_tick(SYMBOL).bid
    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL,
        "position": result.order,
        "price": close_price,
        "deviation": 20,
        "magic": 999999,
        "comment": "Close_Test_Ping",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }
    close_result = mt5.order_send(close_request)
    if close_result.retcode == mt5.TRADE_RETCODE_DONE:
        print("SUCCESS: Test position closed cleanly.")
    else:
        print(f"Close failed: {close_result.retcode} | {close_result.comment}")

mt5.shutdown()