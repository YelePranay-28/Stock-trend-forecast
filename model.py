"""
model.py
========
The "brain" of the application.

What happens here:
  1. Prepare features (X) and target (y) from the enriched DataFrame
  2. Split into train / test sets
  3. Scale features with StandardScaler
  4. Train Linear Regression AND Random Forest
  5. Evaluate both models (RMSE, MAE, R²)
  6. Auto-select the best model
  7. Predict next-day and next-week prices
"""

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from utils import add_technical_indicators, get_feature_columns

#  We cache the trained models per (ticker, period)
#  so re-running the app doesn't re-train every time.

@st.cache_resource(show_spinner=False)
def train_models(ticker: str, df_raw: pd.DataFrame):
    """
    Full ML pipeline: feature engineering → train → evaluate → select best.

    Returns
    -
    dict with keys:
        best_model, best_model_name, scaler,
        feature_cols, results, df_features,
        X_test, y_test, y_pred_best
    """
    # Step 1: Add technical indicators
    df = add_technical_indicators(df_raw)

    feature_cols = get_feature_columns()
    target_col   = "Close"

    # Step 2: Build feature matrix X and target vector y 
    # We predict TOMORROW's closing price.
    # So we shift the target back by 1 day:
    #   X[t]  = today's features
    #   y[t]  = tomorrow's closing price
    df["Target"] = df[target_col].shift(-1)   # next day's close
    df.dropna(subset=["Target"], inplace=True) # remove last row (no future close)

    X = df[feature_cols].values
    y = df["Target"].values

    # Step 3: Train / Test split 
    # We use 80% of data for training, 20% for testing.
    # shuffle=False is CRITICAL for time series — we must not mix future into past!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Step 4: Feature Scaling
    # StandardScaler makes all features have mean=0 and std=1.
    # This helps Linear Regression converge better.
    # We fit ONLY on training data to prevent "data leakage".
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Step 5: Train both models 
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(
            n_estimators=200,     # 200 decision trees
            max_depth=10,         # limit tree depth to avoid overfitting
            random_state=42,
            n_jobs=-1             # use all CPU cores
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        # MAPE = Mean Absolute Percentage Error (easy to interpret as %)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        results[name] = {
            "model":   model,
            "y_pred":  y_pred,
            "RMSE":    round(rmse, 4),
            "MAE":     round(mae,  4),
            "R2":      round(r2,   4),
            "MAPE":    round(mape, 2),
            "Accuracy": round(max(0, 100 - mape), 2),
        }

    # Step 6: Select best model (lowest RMSE wins) 
    best_name  = min(results, key=lambda n: results[n]["RMSE"])
    best_model = results[best_name]["model"]
    y_pred_best = results[best_name]["y_pred"]

    return {
        "best_model":      best_model,
        "best_model_name": best_name,
        "scaler":          scaler,
        "feature_cols":    feature_cols,
        "results":         results,
        "df_features":     df,
        "X_test":          X_test,
        "y_test":          y_test,
        "y_pred_best":     y_pred_best,
    }


def predict_next_prices(trained: dict) -> dict:
    """
    Use the best model to predict:
      • Next day's closing price
      • Prices for the next 5 trading days  (iterative forecasting)
    """

    best_model   = trained["best_model"]
    scaler       = trained["scaler"]
    feature_cols = trained["feature_cols"]
    df           = trained["df_features"]

    # Next-day prediction 
    # Take the LAST row of our feature DataFrame (most recent day)
    last_features  = df[feature_cols].iloc[-1].values.reshape(1, -1)
    last_scaled    = scaler.transform(last_features)
    next_day_price = float(best_model.predict(last_scaled)[0])

    # Next-week prediction (5 trading days) 
    # Strategy: iterative / "rolling" forecast.
    # We predict day+1, then use that prediction as input for day+2, etc.
    # This is approximate — uncertainty compounds each step.
    next_week_prices = []
    current_features = last_features.copy()

    for day in range(5):
        scaled_f   = scaler.transform(current_features)
        pred_price = float(best_model.predict(scaled_f)[0])
        next_week_prices.append(round(pred_price, 2))

        # Shift the feature vector forward by one day
        # (Update Close-related fields with the predicted price)
        col_indices = {col: i for i, col in enumerate(feature_cols)}
        new_features = current_features.copy()

        # Update Lag features using the new predicted price
        if "Lag_1" in col_indices:
            new_features[0][col_indices["Lag_1"]] = pred_price
        if "Lag_5" in col_indices and day >= 4:
            new_features[0][col_indices["Lag_5"]] = pred_price

        current_features = new_features

    return {
        "next_day_price": round(next_day_price, 2),
        "next_week_prices": next_week_prices,
    }
