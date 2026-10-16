# Front End

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data_fetch import get_supported_tickers,fetch_stock_data,fetch_current_price
from model      import train_models, predict_next_prices
from utils      import generate_trading_signal, format_large_number, add_technical_indicators


#PAGE CONFIGURATION
st.set_page_config(
    page_title = "Stock price predictor",
    page_icon = "📈",
    layout = "wide",
    initial_sidebar_state = "expanded"
)
st.title("📈 AI-Powered Stock Price Predictor")
 # SIDEBAR
with st.sidebar:
    st.markdown("### 📈 Stock Predictor")
    st.markdown("---")
    input_mode = st.radio("Select stock by:", 
                          ["📋 Popular list", 
                           "⌨️ Custom ticker"])
    
    tickers_map = get_supported_tickers()

    if input_mode == "📋 Popular list":
        selected_label  = st.selectbox("Choose a stock:", 
                                       list(tickers_map.keys()))
        ticker_symbol = tickers_map[selected_label]
        st.write(f"You selected {ticker_symbol} ")
    else:
        ticker_symbol = st.text_input(
            "Enter Ticker: ",
            value= "AAPL",
            help= "For Indian stocks add '.NS' for NSE or '.BO' for BSE"
        ).strip().upper()
    st.markdown("------")

    period = st.select_slider(
        "Training data period:",
        options=["6mo", "1y", "2y", "3y", "5y"],
        value="2y",
        help="More data = better model, but slower training"
    )
    st.markdown("------")

    predict_btn = st.button("🚀 Predict Now", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("⚠️ Disclaimer: For educational purposes only. Not financial advice.")

if not predict_btn:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🤖 ML Models")           
        st.write("Linear Regression & Random Forest trained on 20+ technical indicators")
    with col2:
        st.subheader("Indicators")
        st.write("RSI, MACD, Bollinger Bands, Moving Averages, ATR, OBV and more")
    with col3:
        st.subheader("Signals")
        st.write("Automated BUY / SELL / HOLD signals with confidence score")
    
    st.info("👈 Select a stock from the sidebar and click **🚀 Predict Now** to begin!")
    st.stop()

with st.spinner(f"Fetching data for {ticker_symbol}"):
    df_raw, error = fetch_stock_data(ticker_symbol, period=period)

live_info = fetch_current_price(ticker_symbol)
company   = live_info.get("company_name")
currency  = live_info.get("currency")

st.markdown(f"{company} - {ticker_symbol}")
h1,h2,h3,h4,h5 = st.columns(5)
h1.metric("current price - ",f"{currency} {live_info.get("price")}")
h2.metric("sector: ",f"{live_info.get("sector")}")
h3.metric("market cap: ",live_info.get("market_cap",0))
h4.metric("Data points: ",f"{len(df_raw)}")
h5.metric("Training period: ",period)

with st.spinner("Training ML models..."):
    trained = train_models(ticker_symbol + "_" + period, df_raw)
predictions = predict_next_prices(trained)

best_model_name = trained["best_model_name"]
results         = trained["results"]
df_feat         = trained["df_features"]
y_test          = trained["y_test"]
y_pred_best     = trained["y_pred_best"]
next_day        = predictions["next_day_price"]
next_week       = predictions["next_week_prices"]

#print(df_feat.head())
#print(results)
signal_info = generate_trading_signal(df_feat, next_day)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predictions & Signal",
    "📊 Price Charts",
    "🤖 Model Performance",
    "📋 Raw Data",
])

with tab1:
    col_pred, col_signal = st.columns([1,1], gap="large")

    with col_pred:
        st.markdown("### 🔮 Price Predictions")
    with col_signal:
        st.markdown("### 📡 Trading Signal")
        fig_gauge = go.Figure(go.Indicator(
            mode   = "gauge+number",
            value  = signal_info["confidence"],
            title  = {"text": "Signal Confidence (%)"},
            gauge  = {
                "axis":  {"range": [0, 100]},
                "bar":   {"color": "#e94551"},
                "steps": [
                    {"range": [0,  40], "color": "#2d3436"},
                    {"range": [40, 70], "color": "#636e72"},
                    {"range": [70,100], "color": "#0f3460"},
                ],
                "threshold": {
                    "line":  {"color": "#e94560", "width": 4},
                    "thickness": 0.75,
                    "value": signal_info["confidence"],
                },
            },
        ))
        fig_gauge.update_layout(
            height          = 220,
            margin          = dict(l=20, r=20, t=40, b=20),
            paper_bgcolor   = "rgba(0,0,0,0)",
            font_color      = "#ffffff",
        )
        st.plotly_chart(fig_gauge, width = "stretch")

        st.markdown("#### 📝 Signal Reasoning")
        for reason in signal_info["reasons"]:
            st.markdown(f"- {reason}")
        
    st.markdown("---")
    st.markdown("### 📈 5-Day Price Forecast Chart")

    last_30_dates  = df_feat.index[-30:].strftime("%Y-%m-%d").tolist()
    last_30_prices = df_feat["Close"].iloc[-30:].tolist()

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=last_30_dates, y=last_30_prices,
        name="Historical Close", line=dict(color="#74b9ff", width=2)
    ))

    fig_forecast.update_layout(
        title       = f"{ticker_symbol} — Historical + 5-Day Forecast",
        xaxis_title = "Date",
        yaxis_title = f"Price ({currency})",
        hovermode   = "x unified",
        height      = 400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(15,15,35,0.8)",
        font_color  = "#ffffff",
        legend      = dict(bgcolor="rgba(0,0,0,0.3)"),
    )
    st.plotly_chart(fig_forecast, width="stretch")