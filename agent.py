# agent.py
import os
import json
import numpy as np
import pandas as pd
import ta
from anthropic import Anthropic

# ── Claude client ────────────────────────────────────────────
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── Load & clean GSE data ────────────────────────────────────
def load_gse_data(filepath="gse_data.csv"):
    df = pd.read_csv(filepath)
    df.columns = [
        "date", "ticker", "year_high", "year_low",
        "prev_close", "open", "last_price", "close",
        "price_change", "bid", "offer", "volume", "value_traded"
    ]
    df["date"]   = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["ticker"] = df["ticker"].str.replace("*", "", regex=False).str.strip()
    df           = df.dropna(subset=["date"])

    num_cols = ["year_high","year_low","prev_close","open","last_price",
                "close","price_change","bid","offer","volume","value_traded"]
    for col in num_cols:
        df[col] = (pd.to_numeric(
            df[col].astype(str).str.replace(",","",regex=False).str.strip()
                   .replace(["","nan","None"], np.nan),
            errors="coerce"))

    df = df[df["close"] > 0].sort_values(["ticker","date"]).reset_index(drop=True)
    return df


def compute_features(df):
    results = []
    for ticker in df["ticker"].unique():
        s = df[df["ticker"] == ticker].copy().sort_values("date")
        if len(s) < 30:
            continue
        c = s["close"]
        s["ma_20"]      = c.rolling(20).mean()
        s["ma_50"]      = c.rolling(50).mean()
        s["ma_200"]     = c.rolling(200).mean()
        s["rsi_14"]     = ta.momentum.RSIIndicator(c, window=14).rsi()
        macd            = ta.trend.MACD(c)
        s["macd_diff"]  = macd.macd_diff()
        bb              = ta.volatility.BollingerBands(c, window=20)
        s["bb_upper"]   = bb.bollinger_hband()
        s["bb_lower"]   = bb.bollinger_lband()
        s["return_1d"]  = c.pct_change(1)   * 100
        s["return_5d"]  = c.pct_change(5)   * 100
        s["return_20d"] = c.pct_change(20)  * 100
        s["return_1y"]  = c.pct_change(252) * 100
        s["vol_ratio"]  = s["volume"] / s["volume"].rolling(20).mean()
        results.append(s)
    return pd.concat(results, ignore_index=True)


def compute_fundamentals(df):
    stats = []
    for ticker, grp in df.groupby("ticker"):
        grp    = grp.sort_values("date")
        close  = grp["close"]
        if len(grp) < 20:
            continue
        last252    = grp.tail(252)
        latest     = grp.iloc[-1]
        ann_vol    = close.pct_change().dropna().std() * (252**0.5) * 100
        avg_value  = grp["value_traded"].tail(20).mean()
        w52_high   = last252["year_high"].max()
        w52_low    = last252["year_low"].min()
        ma50       = grp.tail(50)["close"].mean()
        stats.append({
            "ticker":          ticker,
            "latest_close":    round(latest["close"], 4),
            "latest_date":     str(latest["date"].date()),
            "52w_high":        round(w52_high, 4),
            "52w_low":         round(w52_low,  4),
            "pct_from_high":   round(((latest["close"]-w52_high)/w52_high)*100, 2),
            "pct_from_low":    round(((latest["close"]-w52_low) /w52_low) *100, 2),
            "ann_volatility":  round(ann_vol,   2),
            "avg_daily_value": round(avg_value, 2),
            "ma50":            round(ma50,      4),
            "trend":           "uptrend" if latest["close"] > ma50 else "downtrend",
            "trading_days":    len(grp),
        })
    return pd.DataFrame(stats).sort_values("avg_daily_value", ascending=False)


# ── Load data at startup ─────────────────────────────────────
print("Loading GSE data...")
gse_df         = load_gse_data()
features_df    = compute_features(gse_df)
fundamentals_df = compute_fundamentals(gse_df)
print(f"Ready — {gse_df['ticker'].nunique()} tickers loaded.")


# ── Tool functions ───────────────────────────────────────────
def get_stock_snapshot(ticker):
    ticker = ticker.upper().strip()
    fund   = fundamentals_df[fundamentals_df["ticker"] == ticker]
    tech   = features_df[features_df["ticker"] == ticker].sort_values("date")
    if fund.empty or tech.empty:
        return json.dumps({"error": f"Ticker '{ticker}' not found"})
    f, t = fund.iloc[0], tech.iloc[-1]
    rsi  = t.get("rsi_14")
    def rsi_label(r):
        if pd.isna(r): return "insufficient data"
        if r > 70:     return "overbought"
        if r < 30:     return "oversold"
        return "neutral"
    bb_upper = t.get("bb_upper")
    bb_lower = t.get("bb_lower")
    price    = f["latest_close"]
    if pd.isna(bb_upper) or pd.isna(bb_lower): bb_sig = "insufficient data"
    elif price > bb_upper: bb_sig = "above upper band (overbought)"
    elif price < bb_lower: bb_sig = "below lower band (oversold)"
    else: bb_sig = "within bands"
    return json.dumps({
        "ticker": ticker, "latest_close": price,
        "latest_date": f["latest_date"], "52w_high": f["52w_high"],
        "52w_low": f["52w_low"], "pct_from_high": f["pct_from_high"],
        "trend": f["trend"], "ann_volatility": f["ann_volatility"],
        "avg_daily_value": f["avg_daily_value"],
        "rsi_14": round(rsi, 2) if not pd.isna(rsi) else None,
        "rsi_signal": rsi_label(rsi),
        "macd_signal": "bullish" if not pd.isna(t.get("macd_diff")) and t["macd_diff"] > 0 else "bearish",
        "bb_signal": bb_sig,
        "return_1d":  round(t["return_1d"],  2) if not pd.isna(t["return_1d"])  else None,
        "return_5d":  round(t["return_5d"],  2) if not pd.isna(t["return_5d"])  else None,
        "return_1y":  round(t["return_1y"],  2) if not pd.isna(t["return_1y"])  else None,
        "ma_20": round(t["ma_20"], 4) if not pd.isna(t["ma_20"]) else None,
        "ma_50": round(t["ma_50"], 4) if not pd.isna(t["ma_50"]) else None,
    }, default=str)


def compare_stocks(tickers):
    return json.dumps([
        json.loads(get_stock_snapshot(t.strip()))
        for t in tickers.split(",")
    ], default=str)


def get_top_movers(n=10):
    latest = features_df.sort_values("date").groupby("ticker").last().reset_index()
    latest = latest[latest["return_20d"].notna()]
    return json.dumps({
        "top_gainers": latest.nlargest(n,  "return_20d")[["ticker","close","return_20d","rsi_14"]].to_dict("records"),
        "top_losers":  latest.nsmallest(n, "return_20d")[["ticker","close","return_20d","rsi_14"]].to_dict("records"),
        "as_of_date":  str(latest["date"].max().date()),
    }, default=str)


def get_price_history(ticker, days=90):
    ticker = ticker.upper().strip()
    hist   = gse_df[gse_df["ticker"] == ticker].sort_values("date").tail(days)
    if hist.empty:
        return json.dumps({"error": f"No history for {ticker}"})
    prices = hist["close"].tolist()
    dates  = hist["date"].dt.strftime("%Y-%m-%d").tolist()
    return json.dumps({
        "ticker": ticker, "days_available": len(hist),
        "start_date": dates[0], "end_date": dates[-1],
        "start_price": round(prices[0], 4), "end_price": round(prices[-1], 4),
        "period_return": round(((prices[-1]-prices[0])/prices[0])*100, 2),
        "high": round(max(prices), 4), "low": round(min(prices), 4),
        "recent_closes": [round(p, 4) for p in prices[-7:]],
    }, default=str)


def screen_stocks(min_return_1y=None, max_volatility=None,
                  trend=None, min_daily_value=None, rsi_signal=None):
    latest = features_df.sort_values("date").groupby("ticker").last().reset_index()
    merged = latest.merge(fundamentals_df[["ticker","ann_volatility",
                          "avg_daily_value","trend","pct_from_high"]], on="ticker", how="left")
    def rsi_lbl(r):
        if pd.isna(r): return "unknown"
        if r > 70:     return "overbought"
        if r < 30:     return "oversold"
        return "neutral"
    merged["rsi_signal"] = merged["rsi_14"].apply(rsi_lbl)
    mask = pd.Series([True]*len(merged))
    if min_return_1y   is not None: mask &= merged["return_1y"]       >= min_return_1y
    if max_volatility  is not None: mask &= merged["ann_volatility"]   <= max_volatility
    if trend           is not None: mask &= merged["trend"]            == trend
    if min_daily_value is not None: mask &= merged["avg_daily_value"]  >= min_daily_value
    if rsi_signal      is not None: mask &= merged["rsi_signal"]       == rsi_signal
    screened = merged[mask][["ticker","close","return_1y","ann_volatility",
                              "trend","rsi_14","rsi_signal","avg_daily_value"]]\
               .dropna(subset=["return_1y"]).sort_values("return_1y", ascending=False)
    return json.dumps({"matches": len(screened),
                        "stocks": screened.head(15).to_dict("records")}, default=str)


TOOLS = {
    "get_stock_snapshot": get_stock_snapshot,
    "compare_stocks":     compare_stocks,
    "get_top_movers":     get_top_movers,
    "get_price_history":  get_price_history,
    "screen_stocks":      screen_stocks,
}

TOOL_SCHEMAS = [
    {"name": "get_stock_snapshot",
     "description": "Full technical and price analysis for one GSE stock.",
     "input_schema": {"type":"object","properties":{"ticker":{"type":"string"}},"required":["ticker"]}},
    {"name": "compare_stocks",
     "description": "Compare multiple GSE stocks side by side. Comma-separated tickers.",
     "input_schema": {"type":"object","properties":{"tickers":{"type":"string"}},"required":["tickers"]}},
    {"name": "get_top_movers",
     "description": "Top gaining and losing GSE stocks by 20-day return.",
     "input_schema": {"type":"object","properties":{"n":{"type":"integer"}},"required":[]}},
    {"name": "get_price_history",
     "description": "Recent price history and return summary for a stock.",
     "input_schema": {"type":"object","properties":{
         "ticker":{"type":"string"},"days":{"type":"integer"}},"required":["ticker"]}},
    {"name": "screen_stocks",
     "description": "Screen GSE stocks by return, volatility, trend, liquidity, or RSI.",
     "input_schema": {"type":"object","properties":{
         "min_return_1y":   {"type":"number"},
         "max_volatility":  {"type":"number"},
         "trend":           {"type":"string"},
         "min_daily_value": {"type":"number"},
         "rsi_signal":      {"type":"string"}},"required":[]}},
]

SYSTEM_PROMPT = """You are an expert investment research analyst for the Ghana Stock Exchange (GSE).
You have real historical GSE data from 2007 to 2026 and tools to analyse stocks.

IMPORTANT — you are replying via Telegram so:
- Keep responses concise and well structured
- Use plain text only (no markdown bold/italic — Telegram renders it oddly)
- Use emojis sparingly to add clarity: 📈 bullish, 📉 bearish, ⚠️ risk
- Lead with a one-line verdict, then 3-4 supporting data points, then risks
- Max ~250 words per reply

Always call a tool first before drawing conclusions.
Remind users this is research only, not licensed financial advice."""


# ── Agent runner (one reply per user message) ────────────────
def run_agent(user_message: str, history: list) -> str:
    """
    history: list of prior {role, content} dicts for this user session.
    Returns the agent's reply as a plain string.
    """
    messages  = history + [{"role": "user", "content": user_message}]
    max_steps = 6

    for _ in range(max_steps):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if hasattr(b, "text")), "")

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    fn  = TOOLS.get(block.name)
                    try:
                        vals   = list(block.input.values())
                        result = fn(*vals) if vals else fn()
                    except Exception as e:
                        result = json.dumps({"error": str(e)})
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result,
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",      "content": tool_results})

    return "Sorry, I could not generate a response. Please try again."