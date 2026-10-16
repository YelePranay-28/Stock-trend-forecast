"""
data_fetch.py : 
Responsible for fetching stock data from Yahoo Finance.
Think of this as the "data collector" of our app.
"""

import yfinance as yf
import pandas as pd
import streamlit as st


#  @st.cache_data  →  Streamlit caches the result so we
#  don't re-download the same data on every button click.
#  TTL = 900 seconds = 15 minutes cache lifetime. 


@st.cache_data(ttl=900)
def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Download historical OHLCV data for a given ticker.
    Returns: pd.DataFrame with columns: Open, High, Low, Close, Volume
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            return None, f" No data found for ticker '{ticker}'. Check the symbol."

        # Keep only the columns we need
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index) # ensure datetime index
        df.dropna(inplace=True) # drop any missing rows

        return df, None # (data, error_message)

    except Exception as e:
        return None, f"❌ Error fetching data: {str(e)}"


@st.cache_data(ttl=60)
def fetch_current_price(ticker: str) -> dict:
    """
    Fetch the latest real-time price + basic info for a ticker.
    Returns a dict with keys: price, currency, company_name, change_pct
    """
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        # yfinance exposes different keys depending on the asset type
        price    = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        prev     = info.get("previousClose", price)
        change   = ((price - prev) / prev * 100) if prev else 0

        return {
            "price":        round(price, 2),
            "currency":     info.get("currency", "USD"),
            "company_name": info.get("longName", ticker),
            "change_pct":   round(change, 2),
            "sector":       info.get("sector", "N/A"),
            "market_cap":   info.get("marketCap", 0),
        }
    except Exception:
        return {}


def get_supported_tickers() -> dict:
    """
    Returns a curated dictionary of popular tickers the user can pick from.
    Format:  { "Display Label": "TICKER_SYMBOL" }
    """
    return {
        # 🇺🇸 US Stocks
        "Apple (AAPL)":          "AAPL",
        "Microsoft (MSFT)":      "MSFT",
        "Google (GOOGL)":        "GOOGL",
        "Amazon (AMZN)":         "AMZN",
        "Tesla (TSLA)":          "TSLA",
        "Meta (META)":           "META",
        "NVIDIA (NVDA)":         "NVDA",
        # 🇮🇳 Indian Stocks (NSE)
        "TCS (TCS.NS)":          "TCS.NS",
        "Infosys (INFY.NS)":     "INFY.NS",
        "Reliance (RELIANCE.NS)":"RELIANCE.NS",
        "HDFC Bank (HDFCBANK.NS)":"HDFCBANK.NS",
        "Wipro (WIPRO.NS)":      "WIPRO.NS",
        # Indices
        "S&P 500 ETF (SPY)":     "SPY",
        "NASDAQ ETF (QQQ)":      "QQQ",
    }
