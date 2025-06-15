import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import datetime as dt
import requests  # Changed from yfinance to requests for CoinGecko
import ta
import os
from tensorflow.keras.losses import MeanSquaredError

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Constants ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "btc_lstm_model.h5")
WINDOW_SIZE = 30
FEATURES = ['close', 'SMA_7', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'Signal_Line']
FORECAST_HORIZON = 3  # Predict 3 days ahead

# --- Data Loading Functions ---
@st.cache_data
def load_data():
    file_path = os.path.join(MODEL_DIR, "BTC.csv")
    btc_data = pd.read_csv(file_path)
    btc_data['date'] = pd.to_datetime(btc_data['date'])
    btc_data.set_index('date', inplace=True)
    return btc_data

def fetch_coingecko_data(days=365):
    """Fetch Bitcoin price data from CoinGecko API"""
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date').resample('D').last()
        return df[['close']]
    except Exception as e:
        st.error(f"CoinGecko API error: {e}")
        return pd.DataFrame()

# --- Technical Indicators ---
def add_indicators(df):
    # Calculate all required indicators
    df['SMA_7'] = SMAIndicator(df['close'], window=7).sma_indicator()
    df['EMA_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
    df['EMA_26'] = EMAIndicator(df['close'], window=26).ema_indicator()
    df['RSI'] = RSIIndicator(df['close']).rsi()
    
    bbands = BollingerBands(df['close'])
    df['Bollinger_High'] = bbands.bollinger_hband()
    df['Bollinger_Low'] = bbands.bollinger_lband()
    
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].rolling(window=9).mean()
    
    return df.dropna()

# --- Model Creation ---
def create_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    return model

# --- Main Data Processing ---
btc_data = load_data()
btc_data = add_indicators(btc_data)

# Prepare for 3-day prediction
btc_data['target_close'] = btc_data['close'].shift(-FORECAST_HORIZON)
btc_data = btc_data.dropna()

# --- Scale data ---
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

scaled_data = scaler_features.fit_transform(btc_data[FEATURES])
scaled_target = scaler_target.fit_transform(btc_data[['target_close']])

# --- Create sequences ---
X, y = [], []
for i in range(WINDOW_SIZE, len(scaled_data)):
    X.append(scaled_data[i-WINDOW_SIZE:i])
    y.append(scaled_target[i])
X, y = np.array(X), np.array(y)

# --- Split data ---
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# --- Model Initialization ---
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        st.session_state.model = model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        model = create_model((X_train.shape[1], X_train.shape[2]))
else:
    model = create_model((X_train.shape[1], X_train.shape[2]))

if "model" not in st.session_state:
    with st.spinner("Training model..."):
        history = model.fit(X_train, y_train, 
                          batch_size=32, 
                          epochs=20,
                          validation_data=(X_test, y_test),
                          verbose=0)
        try:
            model.save(MODEL_PATH)
            st.session_state.model = model
            st.success("Model trained and saved successfully!")
        except Exception as e:
            st.error(f"Model save failed: {str(e)}")

# Make predictions
if "model" in st.session_state:
    model = st.session_state.model
    y_pred = model.predict(X_test)
    y_pred = scaler_target.inverse_transform(y_pred)

    y_test = scaler_target.inverse_transform(y_test)
    accuracy = 100 - np.mean(np.abs((y_test - y_pred) / y_test)) * 100
else:
    st.error("Model not available for predictions")
    y_pred = np.zeros_like(y_test)
    accuracy = 0

# --- Streamlit App UI ---
st.title("Bitcoin Price Analysis and Insights")

st.sidebar.header("Sections")
options = st.sidebar.radio("Navigation", 
                          ["Introduction", "Data Overview", "EDA", 
                           "ML Model", "3-Day Prediction", "Conclusion"])

if options == "Introduction":
    st.header("Introduction")
    st.write("""
    Bitcoin (BTC) has emerged as a revolutionary digital asset, transforming the global financial landscape.
    This app analyzes Bitcoin's historical price data and predicts future prices using LSTM neural networks.
    """)
    try:
        image_path = os.path.join(MODEL_DIR, "immm.jpg")
        st.image(image_path, caption="BITCOIN", width=670)
    except:
        st.warning("Could not load header image")

elif options == "Data Overview":
    st.header("Data Overview")
    st.write("### Dataset Columns:")
    st.write("""
    - **open**: Opening price
    - **high**: Highest price
    - **low**: Lowest price  
    - **close**: Closing price
    - **volume**: Trading volume
    - Technical indicators (SMA, EMA, RSI, etc.)
    """)
    
    st.write(f"### Data Summary (Total Rows: {len(btc_data)})")
    st.write(btc_data.describe())
    
    st.write("### Sample Data")
    st.dataframe(btc_data.head(10))

elif options == "EDA":
    st.header("Exploratory Data Analysis")
    
    # Closing Price Plot
    st.write("### Closing Price Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(btc_data['close'], label='Close Price', color='blue')
    ax.set_title('Bitcoin Price History')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)
    
    

elif options == "ML Model":
    st.header("Machine Learning Model Performance")
    
    st.write("### Evaluation Metrics")
    metrics = {
        "Mean Squared Error": mean_squared_error(y_test, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
        "R-squared Score": r2_score(y_test, y_pred),
        "Accuracy (%)": accuracy
    }
    st.table(pd.DataFrame([metrics], index=["Metrics"]))
    
    st.write("### Actual vs Predicted Prices")
    results_df = pd.DataFrame({
        "Date": btc_data.index[-len(y_test):][:10],
        "Actual": y_test.flatten()[:10],
        "Predicted": y_pred.flatten()[:10]
    })
    st.dataframe(results_df)
    
    st.write("### Prediction Visualization")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test[:100], label='Actual', color='blue')
    ax.plot(y_pred[:100], label='Predicted', color='orange', alpha=0.7)
    ax.set_title('Actual vs Predicted Prices')
    ax.legend()
    st.pyplot(fig)

elif options == "3-Day Prediction":
    st.header("3-Day Price Prediction")

    with st.spinner("Fetching latest data..."):
        new_data = fetch_coingecko_data(days=365)

        if new_data.empty:
            st.error("Failed to fetch data")
            st.stop()

        try:
            # Add indicators
            new_data = add_indicators(new_data)

            # Ensure we have the exact same features in the same order
            if not all(feat in new_data.columns for feat in FEATURES):
                missing = [f for f in FEATURES if f not in new_data.columns]
                st.error(f"Missing features: {missing}")
                st.stop()

            latest_data = new_data[FEATURES].tail(WINDOW_SIZE)

            if latest_data.isnull().values.any():
                st.error("Missing values in features")
                st.stop()

            # Scale and predict
            scaled_latest = scaler_features.transform(latest_data)
            sequence = np.expand_dims(scaled_latest, axis=0)
            scaled_pred = model.predict(sequence)

            # Inverse transform
            dummy_array = np.zeros((1, 1))  # Only 1 value to inverse_transform now
            dummy_array[0, 0] = scaled_pred[0][0]
            pred_price = scaler_target.inverse_transform(dummy_array)[0][0]

            pred_date = dt.datetime.today() + dt.timedelta(days=FORECAST_HORIZON)
            st.success(f"Predicted Price for {pred_date.strftime('%Y-%m-%d')}: ${pred_price:.2f}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")


elif options == "Conclusion":
    st.header("Conclusion")
    st.write("""
    This application demonstrates the power of machine learning in financial forecasting.
    Key takeaways:
    - LSTM models can effectively learn Bitcoin price patterns
    - Technical indicators provide valuable signals
    - 3-day predictions show promising accuracy
    """)
    st.write("Future improvements could include:")
    st.write("""
    - Incorporating sentiment analysis
    - Adding more technical indicators
    - Implementing ensemble models
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        <p>Bitcoin Analysis App - Created with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)