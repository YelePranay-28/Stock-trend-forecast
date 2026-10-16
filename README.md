# 📈 AI-Powered Stock Price Predictor

> A portfolio-ready Machine Learning application that predicts stock prices using
> Linear Regression and Random Forest, built with Python + Streamlit.

---

## 🗂️ Project Structure

```
stock_prediction_app/
|
├── app.py              ← 🎨 Streamlit UI (main entry point)
├── model.py            ← 🧠 ML training, evaluation & prediction
├── data_fetch.py       ← 📡 Yahoo Finance data fetching + caching
├── utils.py            ← 🔧 Feature engineering + trading signals
├── requirements.txt    ← 📦 Python dependencies
│
└── .streamlit/
    └── config.toml     ← ⚙️ App theme & server config
```

---

## 🚀 Quick Start (Local Setup)

### Step 1 — Prerequisites
Make sure you have **Python 3.9+** installed.
```bash
python --version    # should print Python 3.9.x or higher
```

### Step 2 — Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
*This installs Streamlit, yfinance, pandas, numpy, scikit-learn, and plotly.*

### Step 4 — Run the app
```bash
streamlit run app.py
```
The app opens automatically at **http://localhost:8501**

---

## 🧠 How the ML Pipeline Works (Beginner Explanation)

```
Yahoo Finance API
       │
       ▼
Raw OHLCV Data (Open, High, Low, Close, Volume)
       │
       ▼ utils.py → add_technical_indicators()
Feature-Engineered DataFrame (23 features)
  ├── Moving Averages (MA_10, MA_20, MA_50)
  ├── Exponential MAs (EMA_12, EMA_26)
  ├── MACD + Signal + Histogram
  ├── RSI (14-day)
  ├── Bollinger Bands (Upper, Lower, Width)
  ├── ATR (volatility)
  ├── OBV (volume pressure)
  ├── Daily Return + 10d Volatility
  └── Lag Features (Lag_1, Lag_5, Lag_10)
       │
       ▼ model.py → train_models()
80/20 Train/Test Split (NO shuffle — time series!)
       │
       ├── StandardScaler (normalise features)
       │
       ├── Linear Regression ──┐
       └── Random Forest ──────┤
                               ▼
                    RMSE / MAE / R² / MAPE
                               │
                    Auto-select best model
                               │
                               ▼
               Next-Day & 5-Day Price Predictions
                               │
                    BUY / SELL / HOLD Signal
```

---

## 📊 Features Explained

| Feature | What it measures | Why it matters |
|---------|-----------------|----------------|
| MA_20 | Average price over 20 days | Smooth out noise, identify trend |
| RSI | Speed of price change (0-100) | Overbought >70, Oversold <30 |
| MACD | EMA_12 minus EMA_26 | Momentum direction |
| Bollinger Bands | Price bands ± 2σ from MA_20 | Volatility & breakout signals |
| ATR | Average daily price range | Market volatility |
| OBV | Cumulative buy/sell volume | Confirms price moves |
| Lag_1 | Yesterday's closing price | Short-term memory |

---

## 🎯 Trading Signal Logic

The signal is based on a **scoring system** (+/- points):

| Condition | Signal | Score |
|-----------|--------|-------|
| Predicted price > current by >1.5% | Bullish | +2 |
| RSI < 30 (oversold) | Bullish | +2 |
| MACD > Signal line | Bullish | +1 |
| Price < Bollinger Lower Band | Bullish | +1 |
| MA_20 > MA_50 | Bullish | +1 |
| (opposite of above) | Bearish | negative |

**Final signal:**
- Score ≥ +3 → **🟢 BUY**
- Score ≤ -3 → **🔴 SELL**
- Otherwise  → **🟡 HOLD**

---

## 📡 Supported Stock Tickers

**US Stocks:** AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA

**Indian Stocks (NSE):** TCS.NS, INFY.NS, RELIANCE.NS, HDFCBANK.NS, WIPRO.NS

**ETFs:** SPY, QQQ

> 💡 For any stock, find its Yahoo Finance ticker and type it in the custom input.
> Indian stocks: add `.NS` for NSE (e.g., `TATAMOTORS.NS`)

---

## 🌐 Deployment on Streamlit Cloud (FREE)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: Stock Price Predictor"
git remote add origin https://github.com/YOUR_USERNAME/stock-predictor.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io**
2. Click **"New app"**
3. Connect your GitHub account
4. Select your repository
5. Set **Main file path** = `app.py`
6. Click **"Deploy!"**

Your app will be live at:
`https://YOUR_USERNAME-stock-predictor-app-XXXX.streamlit.app`

### Step 3 — Share your app
Copy the URL and add it to your **portfolio / LinkedIn / resume**! ✅

---

## 🔧 Alternative Deployment Options

### Render.com (Free tier)
```bash
# Create render.yaml
cat > render.yaml << EOF
services:
  - type: web
    name: stock-predictor
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
EOF
```

### Railway.app
```bash
# railway.json
{
  "build": { "builder": "NIXPACKS" },
  "deploy": {
    "startCommand": "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
  }
}
```

---

## 📱 App Screenshots (Description)

### Screen 1 — Welcome Page
- Dark gradient background (#0f0f23)
- Three feature cards: ML Models | Indicators | Signals
- Sidebar with stock selector and Predict button

### Screen 2 — Prediction Results
- Top row: 5 metric cards (Price, Sector, Market Cap, Data Points, Period)
- Tabs: 🎯 Predictions | 📊 Charts | 🤖 Models | 📋 Data
- Prediction tab: Next-day price, 5-day forecast table, trading signal gauge

### Screen 3 — Charts Tab
- Interactive candlestick chart with MA overlays
- Volume histogram (green/red bars)
- RSI chart with 30/70 reference lines
- MACD with histogram
- Bollinger Bands with price inside band

### Screen 4 — Model Performance Tab
- Metrics table: RMSE, MAE, R², MAPE, Accuracy for both models
- Actual vs Predicted line chart on test data
- Feature Importance horizontal bar chart (Random Forest)

---

## 📝 Interview Talking Points

**Q: What ML models did you use and why?**
> "Linear Regression as a simple baseline, and Random Forest as an ensemble method.
> Random Forest usually wins because it handles non-linear relationships and
> feature interactions better. We auto-select based on RMSE."

**Q: How did you prevent data leakage?**
> "Time-series train/test split with shuffle=False. StandardScaler fitted only on
> training data, then applied to test. Target is shifted by 1 day to predict
> tomorrow's price, not today's."

**Q: What are the limitations?**
> "Stock prices are affected by news, earnings, global events — things no indicator
> captures. Our model works on historical patterns only. It's a learning project, not
> a trading system."

**Q: How did you handle caching?**
> "Streamlit's `@st.cache_data` for data (15 min TTL) and `@st.cache_resource`
> for trained models. This prevents re-training on every button click."

---

## ⚠️ Disclaimer

This project is built for **educational and portfolio purposes only**.
Stock predictions are inherently uncertain. **Do not** use this tool to make
real financial decisions. Always consult a certified financial advisor.

---

*Built with ❤️ using Python, Streamlit, scikit-learn, yfinance, and Plotly*
