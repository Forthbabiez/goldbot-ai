import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="GoldBot AI", page_icon="📈", layout="wide")

st.title("GoldBot AI")
st.caption("XAUUSD 5-minute strategy lab · SIMULATION ONLY")
st.warning("Educational simulator. Uses generated sample prices, not live market data. It does not connect to a broker or place trades.")

with st.sidebar:
    st.header("Backtest settings")
    bars = st.slider("Number of 5-minute candles", 300, 3000, 1000, 100)
    fast_ema = st.number_input("Fast EMA", 3, 100, 20)
    slow_ema = st.number_input("Slow EMA", 5, 200, 50)
    rsi_period = st.number_input("RSI period", 3, 50, 14)
    starting_balance = st.number_input("Virtual starting balance ($)", 100, 100000, 10000, 100)
    risk_pct = st.slider("Virtual risk per trade (%)", 0.1, 2.0, 1.0, 0.1)
    seed = st.number_input("Sample-data seed", 1, 99999, 42)

@st.cache_data
def make_data(n, seed):
    rng = np.random.default_rng(seed)
    # Synthetic prices only: random-walk sample, not a market feed.
    returns = rng.normal(0.00002, 0.0018, n)
    close = 2300 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.15, 1.8, n)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range(end=pd.Timestamp.now().floor("5min"), periods=n, freq="5min")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)

def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

df = make_data(bars, seed)
df["EMA_fast"] = df["Close"].ewm(span=int(fast_ema), adjust=False).mean()
df["EMA_slow"] = df["Close"].ewm(span=int(slow_ema), adjust=False).mean()
df["RSI"] = rsi(df["Close"], int(rsi_period))
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

df["Signal"] = "HOLD"
buy = (df["EMA_fast"] > df["EMA_slow"]) & (df["RSI"].between(50, 70)) & (df["MACD"] > df["MACD_signal"])
sell = (df["EMA_fast"] < df["EMA_slow"]) & (df["RSI"].between(30, 50)) & (df["MACD"] < df["MACD_signal"])
df.loc[buy, "Signal"] = "BUY"
df.loc[sell, "Signal"] = "SELL"

# Educational one-position-at-a-time backtest. Entry at next candle open;
# exit when the opposite signal appears. No spread, slippage, or commissions.
position = 0
entry = None
balance = float(starting_balance)
equity = []
trades = []
risk_fraction = risk_pct / 100

for i in range(1, len(df)):
    signal = df["Signal"].iloc[i-1]
    price = float(df["Open"].iloc[i])
    if position == 0 and signal in ("BUY", "SELL"):
        position = 1 if signal == "BUY" else -1
        entry = price
        entry_time = df.index[i]
        risk_cash = balance * risk_fraction
    elif position != 0 and ((position == 1 and signal == "SELL") or (position == -1 and signal == "BUY")):
        exit_price = price
        pct_move = (exit_price - entry) / entry * position
        pnl = risk_cash * (pct_move / 0.01)  # 1% price move corresponds to 1R in this simplified model
        balance += pnl
        trades.append({"Entry": entry_time, "Exit": df.index[i], "Side": "BUY" if position == 1 else "SELL",
                       "Entry price": entry, "Exit price": exit_price, "P/L ($)": pnl})
        position, entry = 0, None
    equity.append({"Time": df.index[i], "Equity": balance})

trades_df = pd.DataFrame(trades)
equity_df = pd.DataFrame(equity)
if not equity_df.empty:
    equity_values = equity_df["Equity"]
    peak = equity_values.cummax()
    drawdown = ((equity_values - peak) / peak * 100).min()
else:
    drawdown = 0.0

total = len(trades_df)
wins = int((trades_df["P/L ($)"] > 0).sum()) if total else 0
win_rate = wins / total * 100 if total else 0
gross_profit = trades_df.loc[trades_df["P/L ($)"] > 0, "P/L ($)"].sum() if total else 0
gross_loss = abs(trades_df.loc[trades_df["P/L ($)"] < 0, "P/L ($)"].sum()) if total else 0
profit_factor = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0)

a, b, c, d = st.columns(4)
a.metric("Virtual balance", f"${balance:,.2f}", f"{balance-starting_balance:+,.2f}")
b.metric("Closed trades", total)
c.metric("Win rate", f"{win_rate:.1f}%")
d.metric("Max drawdown", f"{drawdown:.2f}%")

st.subheader("XAUUSD · Sample 5-minute candles")
fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                     low=df["Low"], close=df["Close"], name="Sample XAUUSD")])
fig.add_trace(go.Scatter(x=df.index, y=df["EMA_fast"], name=f"EMA {fast_ema}", line=dict(width=1)))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA_slow"], name=f"EMA {slow_ema}", line=dict(width=1)))
fig.update_layout(height=520, xaxis_rangeslider_visible=False, template="plotly_dark",
                  margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Latest strategy reading")
    latest = df.iloc[-1]
    st.metric("Signal", latest["Signal"])
    st.write(f"RSI: **{latest['RSI']:.1f}**" if pd.notna(latest["RSI"]) else "RSI: warming up")
    st.write(f"MACD: **{latest['MACD']:.4f}**")
    st.write(f"MACD signal: **{latest['MACD_signal']:.4f}**")
with right:
    st.subheader("Backtest summary")
    st.write(f"Profit factor: **{profit_factor:.2f}**" if np.isfinite(profit_factor) else "Profit factor: **∞**")
    st.write(f"Gross profit: **${gross_profit:,.2f}**")
    st.write(f"Gross loss: **${gross_loss:,.2f}**")
    st.caption("This simplified test excludes spread, slippage, swaps, commissions and realistic lot sizing.")

if not equity_df.empty:
    st.subheader("Virtual equity curve")
    eqfig = go.Figure()
    eqfig.add_trace(go.Scatter(x=equity_df["Time"], y=equity_df["Equity"], mode="lines", name="Equity"))
    eqfig.update_layout(height=300, template="plotly_dark", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(eqfig, use_container_width=True)

st.subheader("Closed simulated trades")
if total:
    st.dataframe(trades_df.sort_values("Entry", ascending=False), use_container_width=True)
    st.download_button("Download trade history CSV", trades_df.to_csv(index=False), "goldbot_trade_history.csv", "text/csv")
else:
    st.info("No closed simulated trades yet. Try changing the sample-data seed or EMA settings.")

st.caption("Important: All prices are randomly generated demonstration data. A backtest on synthetic data says nothing about future results or real trading performance.")
