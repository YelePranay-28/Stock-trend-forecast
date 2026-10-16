"""
utils.py
Helper functions for:
  -> Feature Engineering  (technical indicators)
  -> Generating trading signals  (BUY / SELL / HOLD)
  -> Formatting numbers for display
"""

import numpy as np
import pandas as pd



#  FEATURE ENGINEERING
#  We add "signals" that traders actually use,
#  so our ML model has richer information.


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a rich set of technical indicators to the OHLCV DataFrame.

    Indicators added
    
    MA_10, MA_20, MA_50   - Simple Moving Averages
    EMA_12, EMA_26        - Exponential Moving Averages
    MACD, MACD_Signal     - Momentum oscillator
    RSI                   - Relative Strength Index  (0-100)
    BB_Upper, BB_Lower    - Bollinger Bands
    ATR                   - Average True Range  (volatility)
    OBV                   - On-Balance Volume
    Daily_Return          - % change from previous day
    """

    d = df.copy()

    # 1. Simple Moving Averages 
    # Average closing price over last N days.
    # When price > MA, the trend is considered bullish.
    d["MA_10"]  = d["Close"].rolling(window=10).mean()
    d["MA_20"]  = d["Close"].rolling(window=20).mean()
    d["MA_50"]  = d["Close"].rolling(window=50).mean()

    # 2. Exponential Moving Averages 
    # Like SMA but gives more weight to recent prices.
    d["EMA_12"] = d["Close"].ewm(span=12, adjust=False).mean()
    d["EMA_26"] = d["Close"].ewm(span=26, adjust=False).mean()

    # 3. MACD (Moving Average Convergence Divergence) 
    # MACD = EMA_12 - EMA_26
    # Signal Line = 9-day EMA of MACD
    # When MACD crosses above Signal → bullish momentum
    d["MACD"]        = d["EMA_12"] - d["EMA_26"]
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"]   = d["MACD"] - d["MACD_Signal"]

    # 4. RSI (Relative Strength Index) 
    # Measures speed & magnitude of price changes.
    # RSI > 70  → overbought (possible sell)
    # RSI < 30  → oversold  (possible buy)
    delta  = d["Close"].diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.rolling(14).mean()
    avg_l  = loss.rolling(14).mean()
    rs     = avg_g / (avg_l + 1e-10)          # avoid division by zero
    d["RSI"] = 100 - (100 / (1 + rs))

    # 5. Bollinger Bands 
    # Upper/Lower bands = MA_20 ± 2 standard deviations
    # Price touching upper band → possible sell
    # Price touching lower band → possible buy
    rolling_std   = d["Close"].rolling(20).std()
    d["BB_Upper"] = d["MA_20"] + 2 * rolling_std
    d["BB_Lower"] = d["MA_20"] - 2 * rolling_std
    d["BB_Width"] = d["BB_Upper"] - d["BB_Lower"]   # volatility proxy

    # 6. ATR (Average True Range) 
    # Measures market volatility regardless of direction
    high_low   = d["High"] - d["Low"]
    high_close = (d["High"] - d["Close"].shift()).abs()
    low_close  = (d["Low"]  - d["Close"].shift()).abs()
    tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d["ATR"]   = tr.rolling(14).mean()

    # 7. OBV (On-Balance Volume) 
    # Cumulative volume that tracks buying vs selling pressure
    obv = [0]
    for i in range(1, len(d)):
        if d["Close"].iloc[i] > d["Close"].iloc[i - 1]:
            obv.append(obv[-1] + d["Volume"].iloc[i])
        elif d["Close"].iloc[i] < d["Close"].iloc[i - 1]:
            obv.append(obv[-1] - d["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    d["OBV"] = obv

    # 8. Daily Return 
    d["Daily_Return"]    = d["Close"].pct_change() * 100
    d["Volatility_10d"]  = d["Daily_Return"].rolling(10).std()

    # 9. Lag features 
    # Give the model yesterday's and last week's close price as context
    d["Lag_1"]  = d["Close"].shift(1)
    d["Lag_5"]  = d["Close"].shift(5)
    d["Lag_10"] = d["Close"].shift(10)

    # Drop rows where indicators couldn't be computed (NaN at the start)
    d.dropna(inplace=True)
    return d



#  TRADING SIGNAL GENERATOR


def generate_trading_signal(df: pd.DataFrame, predicted_price: float) -> dict:
    """
    Generate a BUY / SELL / HOLD signal based on:
      1. Predicted price vs current price
      2. RSI condition
      3. MACD crossover
      4. Bollinger Band position

    Returns a dict with signal, confidence, and reasoning.
    """
    latest      = df.iloc[-1]
    current     = latest["Close"]
    rsi         = latest["RSI"]
    macd        = latest["MACD"]
    macd_sig    = latest["MACD_Signal"]
    bb_upper    = latest["BB_Upper"]
    bb_lower    = latest["BB_Lower"]
    ma_20       = latest["MA_20"]
    ma_50       = latest["MA_50"]

    price_change_pct = (predicted_price - current) / current * 100

    # Score system: positive = bullish, negative = bearish
    score = 0
    reasons = []

    # Rule 1: Predicted price direction 
    if price_change_pct > 0.2:
        score += 1
        reasons.append(f"📈 Model predicts +{price_change_pct:.1f}% price rise")
    elif price_change_pct < -0.2:
        score -= 1
        reasons.append(f"📉 Model predicts {price_change_pct:.1f}% price drop")
    else:
        reasons.append(f"↔️ Model predicts small change ({price_change_pct:.1f}%)")

    # Rule 2: RSI 
    if rsi < 45:
        score += 1
        reasons.append(f"💚 RSI={rsi:.1f} → Oversold (potential bounce)")
    elif rsi > 55:
        score -= 1
        reasons.append(f"🔴 RSI={rsi:.1f} → Overbought (potential pullback)")
    else:
        reasons.append(f"⚪ RSI={rsi:.1f} → Neutral zone")

    # Rule 3: MACD 
    if macd > macd_sig:
        score += 1
        reasons.append("💚 MACD above signal line → Bullish momentum")
    else:
        score -= 1
        reasons.append("🔴 MACD below signal line → Bearish momentum")

    # Rule 4: Bollinger Bands 
    bb_width = bb_upper - bb_lower
    if bb_width >0 :
        if current < bb_lower + bb_width*0.25:
            score += 1
            reasons.append("💚 Price below lower Bollinger Band → Possible reversal up")
        elif current > bb_upper - bb_width*0.25:
            score -= 1
            reasons.append("🔴 Price above upper Bollinger Band → Possible reversal down")

    # Rule 5: Moving Average trend 
    if ma_20 > ma_50:
        score += 2
        reasons.append("💚 MA20 > MA50 → Short-term uptrend")
    else:
        score -= 2
        reasons.append("🔴 MA20 < MA50 → Short-term downtrend")

    # Translate score to signal 
    if score >= 2:
        signal     = "BUY"
        emoji      = "🟢"
        confidence = min(50 + score * 7, 92)
    elif score <= -2:
        signal     = "SELL"
        emoji      = "🔴"
        confidence = min(50 + abs(score) * 7, 92)
    else:
        signal     = "HOLD"
        emoji      = "🟡"
        confidence = 50 + abs(score) * 5

    return {
        "signal":     signal,
        "emoji":      emoji,
        "score":      score,
        "confidence": confidence,
        "reasons":    reasons,
        "price_change_pct": price_change_pct,
    }



#  FORMATTING HELPERS


def format_large_number(n: float) -> str:
    """Format large numbers: 1500000 → '1.50M'"""
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    return f"{n:,.0f}"


def get_feature_columns() -> list:
    """Return the list of feature columns the ML model trains on."""
    return [
        "Open", "High", "Low", "Volume",
        "MA_10", "MA_20", "MA_50",
        "EMA_12", "EMA_26",
        "MACD", "MACD_Signal", "MACD_Hist",
        "RSI",
        "BB_Upper", "BB_Lower", "BB_Width",
        "ATR", "OBV",
        "Daily_Return", "Volatility_10d",
        "Lag_1", "Lag_5", "Lag_10",
    ]
