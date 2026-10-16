"""
app.py  -  Main Streamlit Application

This is the entry point. Run with:
    streamlit run app.py

UI sections:
  1. Sidebar  - stock selector, period, controls
  2. Header   - company info + live price card
  3. Tab 1    - Predictions + Trading Signal
  4. Tab 2    - Historical Price Chart + Technical Indicators
  5. Tab 3    - Model Performance (RMSE / MAE / R²)
  6. Tab 4    - Raw Data Table
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Our own modules 
from data_fetch import fetch_stock_data, fetch_current_price, get_supported_tickers
from model      import train_models, predict_next_prices
from utils      import generate_trading_signal, format_large_number, add_technical_indicators


#  PAGE CONFIGURATION
 
st.set_page_config(
    page_title  = "📈 Stock Price Predictor",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


#  CUSTOM CSS  – Makes the app look polished

st.markdown("""
<style>
/* ── Global font & background ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

/* ── Signal badge ── */
.signal-buy  { background:#00b894; color:#fff; padding:8px 20px;
               border-radius:20px; font-weight:700; font-size:1.2rem; }
.signal-sell { background:#d63031; color:#fff; padding:8px 20px;
               border-radius:20px; font-weight:700; font-size:1.2rem; }
.signal-hold { background:#fdcb6e; color:#333; padding:8px 20px;
               border-radius:20px; font-weight:700; font-size:1.2rem; }

/* ── Info box ── */
.info-card {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    border-left: 4px solid #e94560;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%); }
</style>
""", unsafe_allow_html=True)


#  SIDEBAR
 
with st.sidebar:
    st.markdown("## 📈 Stock Predictor")
    st.markdown("---")

    input_mode = st.radio("Select stock by:", [" Popular list", "⌨ Custom ticker"])

    tickers_map = get_supported_tickers()

    if input_mode == " Popular list":
        selected_label  = st.selectbox("Choose a stock:", list(tickers_map.keys()))
        ticker_symbol   = tickers_map[selected_label]
    else:
        ticker_symbol = st.text_input(
            "Enter ticker (e.g. AAPL, TCS.NS):",
            value="AAPL",
            help="For Indian stocks add '.NS' for NSE or '.BO' for BSE"
        ).strip().upper()

    st.markdown("---")

    # Historical data period 
    period = st.select_slider(
        "Training data period:",
        options=["6mo", "1y", "2y", "3y", "5y"],
        value="2y",
        help="More data = better model, but slower training"
    )

    # Show which model won 
    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    1. **Fetch** historical data from Yahoo Finance
    2. **Engineer** 20+ technical features (RSI, MACD, Bollinger Bands…)
    3. **Train** Linear Regression & Random Forest
    4. **Auto-select** the best model (lowest RMSE)
    5. **Predict** next day & next week prices
    6. **Generate** BUY / SELL / HOLD signal
    """)

    
    if st.button("🚀 Predict Now"):
        st.session_state["predicted"] = True
    predict_btn = st.session_state.get("predicted", False)
    st.markdown("---")
    st.caption("⚠️ Disclaimer: For educational purposes only. Not financial advice.")



#  MAIN CONTENT

st.title("📈 AI-Powered Stock Price Predictor")
st.markdown("*Machine Learning meets the Stock Market — Linear Regression & Random Forest*")

if not predict_btn:
    # Welcome screen 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='info-card'>            
        <h4> ML Models</h4>           
        <p>Linear Regression & Random Forest trained on 20+ technical indicators</p>
        </div>""",unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='info-card'>
        <h4>📊 Indicators</h4>
        <p>RSI, MACD, Bollinger Bands, Moving Averages, ATR, OBV and more</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='info-card'>
        <h4>🎯 Signals</h4>
        <p>Automated BUY / SELL / HOLD signals with confidence score</p>
        </div>""", unsafe_allow_html=True)

    st.info("👈 Select a stock from the sidebar and click **🚀 Predict Now** to begin!")
    st.stop()



#  DATA FETCHING

with st.spinner(f"📡 Fetching data for **{ticker_symbol}**…"):
    df_raw, error = fetch_stock_data(ticker_symbol, period=period)

if error:
    st.error(error)
    st.stop()

if df_raw is None or len(df_raw) < 100:
    st.error(f"Not enough historical data for {ticker_symbol}. Try a different ticker or period.")
    st.stop()

# Live price info 
live_info = fetch_current_price(ticker_symbol)
company   = live_info.get("company_name", ticker_symbol)
currency  = live_info.get("currency", "USD")



#  COMPANY HEADER

st.markdown(f"## {company}  `{ticker_symbol}`")

h1, h2, h3, h4, h5 = st.columns(5)
h1.metric("💰 Current Price",
          f"{currency} {live_info.get('price', 'N/A')}",
          f"{live_info.get('change_pct', 0):+.2f}%")
h2.metric("🏭 Sector",       live_info.get("sector", "N/A"))
h3.metric("📊 Market Cap",   format_large_number(live_info.get("market_cap", 0)))
h4.metric("📅 Data Points",  f"{len(df_raw):,}")
h5.metric("📆 Training Period", period)



#  MODEL TRAINING

with st.spinner("🧠 Training ML models (Linear Regression + Random Forest)…"):
    # Pass ticker as cache key so each stock gets its own cached model
    trained = train_models(ticker_symbol + "_" + period, df_raw)

predictions = predict_next_prices(trained)

best_model_name = trained["best_model_name"]
results         = trained["results"]
df_feat         = trained["df_features"]
y_test          = trained["y_test"]
y_pred_best     = trained["y_pred_best"]
next_day        = predictions["next_day_price"]
next_week       = predictions["next_week_prices"]

# Generate trading signal 
signal_info = generate_trading_signal(df_feat, next_day)



#  TABS

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predictions & Signal",
    "📊 Price Charts",
    "🤖 Model Performance",
    "📋 Raw Data",
])



#  TAB 1: PREDICTIONS & SIGNAL

with tab1:
    col_pred, col_signal = st.columns([1, 1], gap="large")

    # Prediction panel
    with col_pred:
        st.markdown("### 🔮 Price Predictions")
        current_price = df_feat["Close"].iloc[-1]

        p1, p2 = st.columns(2)
        p1.metric(
            "📅 Next Day",
            f"{currency} {next_day:,.2f}",
            f"{(next_day - current_price)/current_price*100:+.2f}%"
        )
        p2.metric(
            "📆 Next 5 Days (avg)",
            f"{currency} {np.mean(next_week):,.2f}",
            f"{(np.mean(next_week) - current_price)/current_price*100:+.2f}%"
        )

        # Next-week forecast table
        st.markdown("#### 📅 5-Day Forecast")
        today = datetime.now()
        forecast_dates = []
        d = today
        while len(forecast_dates) < 5:
            d += timedelta(days=1)
            if d.weekday() < 5:          # skip weekends
                forecast_dates.append(d.strftime("%a %b %d"))

        forecast_df = pd.DataFrame({
            "Trading Day":      forecast_dates,
            "Predicted Close":  [f"{currency} {p:,.2f}" for p in next_week],
            "Change from Today":[f"{(p-current_price)/current_price*100:+.2f}%" for p in next_week],
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        # Best model badge 
        st.success(f"🏆 Best Model: **{best_model_name}**  |  "
                   f"Accuracy: **{results[best_model_name]['Accuracy']:.1f}%**  |  "
                   f"RMSE: **{results[best_model_name]['RMSE']:.2f}**")

    # Signal panel 
    with col_signal:
        st.markdown("### 📡 Trading Signal")

        sig     = signal_info["signal"]
        css_cls = f"signal-{sig.lower()}"
        st.markdown(
            f"<div style='text-align:center; margin:10px 0'>"
            f"<span class='{css_cls}'>{signal_info['emoji']} {sig}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Confidence gauge using a Plotly indicator
        fig_gauge = go.Figure(go.Indicator(
            mode   = "gauge+number",
            value  = signal_info["confidence"],
            title  = {"text": "Signal Confidence (%)"},
            gauge  = {
                "axis":  {"range": [0, 100]},
                "bar":   {"color":"#e94551"},
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
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("#### 📝 Signal Reasoning")
        for reason in signal_info["reasons"]:
            st.markdown(f"- {reason}")

    # Next-week forecast chart
    st.markdown("---")
    st.markdown("### 📈 5-Day Price Forecast Chart")

    last_30_dates  = df_feat.index[-30:].strftime("%Y-%m-%d").tolist()
    last_30_prices = df_feat["Close"].iloc[-30:].tolist()

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=last_30_dates, y=last_30_prices,
        name="Historical Close", line=dict(color="#74b9ff", width=2)
    ))

    forecast_x = [last_30_dates[-1]] + forecast_dates
    forecast_y = [last_30_prices[-1]] + next_week
    fig_forecast.add_trace(go.Scatter(
        x=forecast_x, y=forecast_y,
        name="Predicted", line=dict(color="#e94560", width=2, dash="dash"),
        mode="lines+markers", marker=dict(size=8)
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
    st.plotly_chart(fig_forecast, use_container_width=True)

#  TAB 2: PRICE CHARTS & TECHNICAL INDICATORS

with tab2:
    # Candlestick + Volume
    st.markdown("### 🕯️ Candlestick Chart + Volume")
    display_days = st.slider("Show last N trading days:", 60, len(df_raw), 180)
    df_chart = df_raw.iloc[-display_days:]

    fig_candle = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=("Price (Candlestick)", "Volume")
    )
    fig_candle.add_trace(go.Candlestick(
        x=df_chart.index, open=df_chart["Open"],
        high=df_chart["High"], low=df_chart["Low"], close=df_chart["Close"],
        name="OHLC",
        increasing_line_color="#00b84d",
        decreasing_line_color="#d63031",
    ), row=1, col=1)

    # Overlay MA_20 and MA_50 on candlestick
    df_feat_chart = df_feat.iloc[-display_days:]
    fig_candle.add_trace(go.Scatter(
        x=df_feat_chart.index, y=df_feat_chart["MA_20"],
        name="MA 20", line=dict(color="#fdcb6e", width=1.5)
    ), row=1, col=1)
    fig_candle.add_trace(go.Scatter(
        x=df_feat_chart.index, y=df_feat_chart["MA_50"],
        name="MA 50", line=dict(color="#e17055", width=1.5)
    ), row=1, col=1)

    # Volume bars
    colors = ["#00b894" if c >= o else "#d63031"
              for c, o in zip(df_chart["Close"], df_chart["Open"])]
    fig_candle.add_trace(go.Bar(
        x=df_chart.index, y=df_chart["Volume"],
        name="Volume", marker_color=colors, opacity=0.7
    ), row=2, col=1)

    fig_candle.update_layout(
        height=600, xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(15,15,35,0.8)",
        font_color="white", showlegend=True,
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # Technical Indicators grid
    st.markdown("### 📉 Technical Indicators")
    ti_col1, ti_col2 = st.columns(2)

    with ti_col1:
        # RSI chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df_feat_chart.index, y=df_feat_chart["RSI"],
            fill="tonexty", fillcolor="rgba(116,185,255,0.1)",
            line=dict(color="#74b9ff"), name="RSI"
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#d63031",
                          annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00b894",
                          annotation_text="Oversold (30)")
        fig_rsi.update_layout(
            title="RSI (14)", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,35,0.8)",
            font_color="white", yaxis=dict(range=[0,100])
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

    with ti_col2:
        # MACD chart
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=df_feat_chart.index, y=df_feat_chart["MACD"],
            name="MACD", line=dict(color="#74b9ff")
        ))
        fig_macd.add_trace(go.Scatter(
            x=df_feat_chart.index, y=df_feat_chart["MACD_Signal"],
            name="Signal", line=dict(color="#e17055")
        ))
        fig_macd.add_trace(go.Bar(
            x=df_feat_chart.index, y=df_feat_chart["MACD_Hist"],
            name="Histogram",
            marker_color=["#00b894" if v >= 0 else "#d63031"
                          for v in df_feat_chart["MACD_Hist"]]
        ))
        fig_macd.update_layout(
            title="MACD", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,35,0.8)",
            font_color="white"
        )
        st.plotly_chart(fig_macd, use_container_width=True)

    # Bollinger Bands 
    st.markdown("#### 📐 Bollinger Bands")
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(
        x=df_feat_chart.index, y=df_feat_chart["BB_Upper"],
        name="Upper Band", line=dict(color="#e17055", dash="dash")
    ))
    fig_bb.add_trace(go.Scatter(
        x=df_feat_chart.index, y=df_feat_chart["Close"],
        name="Close", line=dict(color="#74b9ff", width=2),
        fill="tonexty", fillcolor="rgba(116,185,255,0.05)"
    ))
    fig_bb.add_trace(go.Scatter(
        x=df_feat_chart.index, y=df_feat_chart["BB_Lower"],
        name="Lower Band", line=dict(color="#00b894", dash="dash"),
        fill="tonexty", fillcolor="rgba(0,184,148,0.05)"
    ))
    fig_bb.add_trace(go.Scatter(
        x=df_feat_chart.index, y=df_feat_chart["MA_20"],
        name="MA 20", line=dict(color="#fdcb6e", width=1)
    ))
    fig_bb.update_layout(
        height=350, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,35,0.8)", font_color="white"
    )
    st.plotly_chart(fig_bb, use_container_width=True)

#  TAB 3: MODEL PERFORMANCE

with tab3:
    st.markdown("### 🤖 Model Comparison")
    metrics_data = []
    # Metrics table
    for name, res in results.items():
        metrics_data.append({
            "Model":    name,
            "RMSE ↓":   res["RMSE"],          # ↓ = lower is better
            "MAE ↓":    res["MAE"],
            "R² ↑":     res["R2"],             # ↑ = higher is better
            "MAPE% ↓":  res["MAPE"],
            "Accuracy": f"{res['Accuracy']:.1f}%",
            "🏆 Best":   "✅" if name == best_model_name else "",
        })

    st.dataframe(
        pd.DataFrame(metrics_data),
        use_container_width=True, hide_index=True
    )
    st.caption("RMSE = Root Mean Squared Error | MAE = Mean Absolute Error | R² = Coefficient of Determination | MAPE = Mean Absolute Percentage Error")

    # Actual vs Predicted chart 
    st.markdown("### 📊 Actual vs Predicted (Test Set)")

    n_test = len(y_test)
    test_indices = df_feat.index[-n_test:]

    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(
        x=test_indices, y=y_test,
        name="Actual Price", line=dict(color="#74b9ff", width=2)
    ))
    fig_avp.add_trace(go.Scatter(
        x=test_indices, y=y_pred_best,
        name=f"Predicted ({best_model_name})",
        line=dict(color="#e94560", width=2, dash="dot")
    ))
    fig_avp.update_layout(
        title   = "Actual vs Predicted Prices on Test Data",
        height  = 400, hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(15,15,35,0.8)",
        font_color="white",
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    # Bar chart of metrics
    m1, m2 = st.columns(2)
    with m1:
        fig_rmse = go.Figure(go.Bar(
            x=list(results.keys()),
            y=[r["RMSE"] for r in results.values()],
            marker_color=["#e94560", "#74b9ff"],
            text=[f"{r['RMSE']:.2f}" for r in results.values()],
            textposition="outside",
        ))
        fig_rmse.update_layout(
            title="RMSE Comparison (lower = better)", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,35,0.8)",
            font_color="white"
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    with m2:
        fig_r2 = go.Figure(go.Bar(
            x=list(results.keys()),
            y=[r["R2"] for r in results.values()],
            marker_color=["#e94560", "#74b9ff"],
            text=[f"{r['R2']:.4f}" for r in results.values()],
            textposition="outside",
        ))
        fig_r2.update_layout(
            title="R² Score (higher = better)", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,35,0.8)",
            font_color="white"
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    # Feature importance (only Random Forest supports this)
    if "Random Forest" in results:
        st.markdown("### 🌟 Feature Importance (Random Forest)")
        rf_model      = results["Random Forest"]["model"]
        importances   = rf_model.feature_importances_
        feat_imp_df   = pd.DataFrame({
            "Feature":    trained["feature_cols"],
            "Importance": importances,
        }).sort_values("Importance", ascending=False).head(15)

        fig_fi = px.bar(
            feat_imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Reds",
            title="Top 15 Most Important Features"
        )
        fig_fi.update_layout(
            height=450, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,35,0.8)", font_color="white",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_fi, use_container_width=True)

 
#  TAB 4: RAW DATA
 
with tab4:
    st.markdown("### 📋 Historical Data (Last 100 rows)")
    display_df = df_raw.tail(100).copy()
    display_df.index = display_df.index.strftime("%Y-%m-%d")
    st.dataframe(display_df.style.format("{:.2f}"), use_container_width=True)

    st.markdown("### 🔬 Feature-Engineered Data (Last 50 rows)")
    feat_display = df_feat[
        ["Close", "MA_20", "MA_50", "RSI", "MACD", "BB_Upper", "BB_Lower", "ATR"]
    ].tail(50).copy()
    feat_display.index = feat_display.index.strftime("%Y-%m-%d")
    st.dataframe(feat_display.style.format("{:.2f}"), use_container_width=True)

    # Download button 
    csv = df_raw.to_csv()
    st.download_button(
        label     = "⬇️ Download Full Historical Data (CSV)",
        data      = csv,
        file_name = f"{ticker_symbol}_historical.csv",
        mime      = "text/csv",
    )


#  FOOTER

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#636e72; font-size:0.85rem'>"
    "⚠️ <b>Disclaimer:</b> This tool is for <b>educational purposes only</b>. "
    "Stock predictions are inherently uncertain. Do not make financial decisions "
    "based solely on this tool. Always consult a certified financial advisor."
    "</div>",
    unsafe_allow_html=True
)
