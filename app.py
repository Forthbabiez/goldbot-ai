import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="GoldBot AI", page_icon="📈", layout="wide")

API_URL = "https://xaus.com/api/v1/chart"

st.title("GoldBot AI")
st.caption("XAU/USD chart lab · 5-minute candles · EDUCATIONAL SIMULATOR ONLY")
st.warning(
    "This app displays market data for learning and simulated analysis only. "
    "It does not connect to a broker, give personalised financial advice, or place orders. "
    "The chart feed may differ from your broker's XAUUSD quote and may be delayed."
)

with st.sidebar:
    st.header("Chart settings")
    period = st.selectbox("History window", ["1d", "5d", "1mo", "3mo"], index=1)
    refresh = st.button("Refresh market data", use_container_width=True)
    st.divider()
    st.header("Indicator settings")
    fast_ema = st.number_input("Fast EMA", 3, 100, 20)
    slow_ema = st.number_input("Slow EMA", 5, 200, 50)
    rsi_period = st.number_input("RSI period", 3, 50, 14)
    st.divider()
    st.header("Paper-test settings")
    starting_balance = st.number_input("Virtual starting balance ($)", 100, 100000, 10000, 100)
    risk_pct = st.slider("Virtual risk setting (%)", 0.1, 2.0, 1.0, 0.1)

if refresh:
    st.cache_data.clear()

@st.cache_data(ttl=60, show_spinner=False)
def load_market_data(history_window):
    params = {"symbol": "xau", "range": history_window, "interval": "5m"}
    response = requests.get(API_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()

    # The feed's documented chart response contains OHLCV points:
    # t = Unix timestamp; o/h/l/c = open/high/low/close; v = volume.
    points = payload.get("points") if isinstance(payload, dict) else None
    if points is None and isinstance(payload, dict):
        points = payload.get("data")
    if not points:
        message = payload.get("message", "The provider returned no candle data.") if isinstance(payload, dict) else "Unexpected data format."
        raise ValueError(message)

    frame = pd.DataFrame(points)
    required = {"t", "o", "h", "l", "c"}
    if not required.issubset(frame.columns):
        raise ValueError("The provider response did not contain the expected OHLC candle fields.")

    frame = frame.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    frame["Time"] = pd.to_datetime(pd.to_numeric(frame["t"]), unit="s", utc=True)
    for col in ["Open", "High", "Low", "Close"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["Time", "Open", "High", "Low", "Close"])
    frame = frame.drop_duplicates(subset=["Time"]).sort_values("Time").set_index("Time")
    frame.index = frame.index.tz_convert("Africa/Johannesburg")
    if len(frame) < 60:
        raise ValueError(f"Only {len(frame)} candles were returned. Try refreshing or choosing a wider history window.")
    return frame

try:
    with st.spinner("Loading real XAU/USD 5-minute candles..."):
        df = load_market_data(period)
except Exception as exc:
    st.error(f"Could not load market data: {exc}")
    st.info("No simulated prices are substituted. Try Refresh market data again later.")
    st.stop()

df["EMA_fast"] = df["Close"].ewm(span=int(fast_ema), adjust=False).mean()
df["EMA_slow"] = df["Close"].ewm(span=int(slow_ema), adjust=False).mean()

def calculate_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

df["RSI"] = calculate_rsi(df["Close"], int(rsi_period))
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

df["Signal"] = "HOLD"
buy_condition = (
    (df["EMA_fast"] > df["EMA_slow"])
    & df["RSI"].between(50, 70)
    & (df["MACD"] > df["MACD_signal"])
)
sell_condition = (
    (df["EMA_fast"] < df["EMA_slow"])
    & df["RSI"].between(30, 50)
    & (df["MACD"] < df["MACD_signal"])
)
df.loc[buy_condition, "Signal"] = "BUY"
df.loc[sell_condition, "Signal"] = "SELL"

latest = df.iloc[-1]
last_time = df.index[-1]
m1, m2, m3 = st.columns(3)
m1.metric("Latest candle close", f"${latest['Close']:,.2f}")
m2.metric("Latest candle time (SAST)", last_time.strftime("%d %b %Y · %H:%M"))
m3.metric("Candles loaded", f"{len(df):,}")

st.caption(
    "Data source: XAUS chart API (documented as a Yahoo Finance proxy). "
    "Candles are 5-minute XAU/USD data, not a guaranteed match for any broker's CFD feed."
)

# Educational one-position-at-a-time paper test. Signals are evaluated on the
# previous candle and entries/exits are simulated at the next candle open.
position = 0
entry_price = None
entry_time = None
balance = float(starting_balance)
risk_cash = 0.0
trades = []
equity = []

for i in range(1, len(df)):
    prior_signal = df["Signal"].iloc[i - 1]
    next_open = float(df["Open"].iloc[i])

    if position == 0 and prior_signal in ("BUY", "SELL"):
        position = 1 if prior_signal == "BUY" else -1
        entry_price = next_open
        entry_time = df.index[i]
        risk_cash = balance * (risk_pct / 100)
    elif position != 0 and (
        (position == 1 and prior_signal == "SELL")
        or (position == -1 and prior_signal == "BUY")
    ):
        pct_move = ((next_open - entry_price) / entry_price) * position
        pnl = risk_cash * (pct_move / 0.01)
        balance += pnl
        trades.append({
            "Entry": entry_time,
            "Exit": df.index[i],
            "Side": "BUY" if position == 1 else "SELL",
            "Entry price": entry_price,
            "Exit price": next_open,
            "P/L ($)": pnl,
        })
        position = 0
        entry_price = None
        entry_time = None
    equity.append({"Time": df.index[i], "Equity": balance})

trades_df = pd.DataFrame(trades)
equity_df = pd.DataFrame(equity)
total = len(trades_df)
wins = int((trades_df["P/L ($)"] > 0).sum()) if total else 0
win_rate = wins / total * 100 if total else 0.0
gross_profit = float(trades_df.loc[trades_df["P/L ($)"] > 0, "P/L ($)"].sum()) if total else 0.0
gross_loss = abs(float(trades_df.loc[trades_df["P/L ($)"] < 0, "P/L ($)"].sum())) if total else 0.0
profit_factor = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
if not equity_df.empty:
    peak = equity_df["Equity"].cummax()
    drawdown = float(((equity_df["Equity"] - peak) / peak * 100).min())
else:
    drawdown = 0.0

st.subheader("XAU/USD · Real 5-minute candles")
fig = go.Figure(data=[go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="XAU/USD"
)])
fig.add_trace(go.Scatter(x=df.index, y=df["EMA_fast"], name=f"EMA {fast_ema}", line=dict(width=1)))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA_slow"], name=f"EMA {slow_ema}", line=dict(width=1)))
fig.update_layout(
    height=520, xaxis_rangeslider_visible=False, template="plotly_dark",
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Indicator reading")
    st.metric("Rule-based condition", str(latest["Signal"]))
    st.write(f"RSI: **{latest['RSI']:.1f}**" if pd.notna(latest["RSI"]) else "RSI: warming up")
    st.write(f"MACD: **{latest['MACD']:.4f}**")
    st.write(f"MACD signal: **{latest['MACD_signal']:.4f}**")
    st.caption("These are indicator conditions for study, not instructions to trade.")
with right:
    st.subheader("Historical paper-test results")
    st.metric("Virtual balance", f"${balance:,.2f}", f"{balance-starting_balance:+,.2f}")
    st.metric("Closed simulated trades", total)
    st.metric("Win rate", f"{win_rate:.1f}%")
    st.metric("Maximum drawdown", f"{drawdown:.2f}%")
    st.write(f"Profit factor: **{profit_factor:.2f}**" if np.isfinite(profit_factor) else "Profit factor: **∞**")

st.caption(
    "Limitations: this simple paper test excludes spread, slippage, swaps, commissions, "
    "market gaps and realistic position sizing. An open position at the end of the data "
    "window is not included in closed-trade results. Past test results do not predict future outcomes."
)

if not equity_df.empty:
    st.subheader("Virtual equity curve")
    eqfig = go.Figure()
    eqfig.add_trace(go.Scatter(x=equity_df["Time"], y=equity_df["Equity"], mode="lines", name="Virtual equity"))
    eqfig.update_layout(height=300, template="plotly_dark", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(eqfig, use_container_width=True)

st.subheader("Closed simulated trades")
if total:
    st.dataframe(trades_df.sort_values("Entry", ascending=False), use_container_width=True)
    st.download_button(
        "Download paper-test history CSV",
        trades_df.to_csv(index=False),
        "goldbot_paper_test_history.csv",
        "text/csv",
    )
else:
    st.info("No closed simulated trades were found in this selected history window.")

st.caption("GoldBot AI is a learning and chart-analysis project. It cannot execute real trades.")
