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
import yfinance as yf
import ta
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Model Path Setup ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "btc_lstm_model.h5")

# --- Data Loading ---
@st.cache_data
def load_data():
    file_path = os.path.join(MODEL_DIR, "BTC.csv")
    btc_data = pd.read_csv(file_path)
    btc_data['date'] = pd.to_datetime(btc_data['date'])
    btc_data.set_index('date', inplace=True)
    return btc_data

btc_data = load_data()

# --- Technical Indicators ---
def add_indicators(df):
    df['SMA_7'] = SMAIndicator(df['close'], window=7).sma_indicator()
    df['EMA_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
    df['EMA_26'] = EMAIndicator(df['close'], window=26).ema_indicator()
    df['RSI'] = RSIIndicator(df['close']).rsi()
    bbands = BollingerBands(df['close'])
    df['Bollinger_High'] = bbands.bollinger_hband()
    df['Bollinger_Low'] = bbands.bollinger_lband()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].rolling(window=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    return df.dropna()

btc_data = add_indicators(btc_data)

# --- Prepare Data for 3-Day Prediction ---
features = ['close', 'SMA_7', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'Signal_Line']
target = 'close'
forecast_horizon = 3  # Predict 3 days ahead

# Shift target for 3-day prediction
btc_data['target_close'] = btc_data[target].shift(-forecast_horizon)
btc_data = btc_data.dropna()

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(btc_data[features])
scaled_target = scaler.fit_transform(btc_data[['target_close']])

# Create sequences
window_size = 30
X, y = [], []
for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i])
    y.append(scaled_target[i])
X, y = np.array(X), np.array(y)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- Model Setup ---
def create_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        st.session_state.model = model
        st.success("Loaded pre-trained model")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        model = create_model()
else:
    model = create_model()

if "model" not in st.session_state:
    with st.spinner("Training model..."):
        model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)
        model.save(MODEL_PATH)
        st.session_state.model = model
        st.success("Model trained and saved")

model = st.session_state.model

# --- Predictions ---
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)
accuracy = 100 - np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# --- Streamlit App ---
st.title("Bitcoin Price Analysis and Insights")

st.sidebar.header("Sections")
options = st.sidebar.radio("Sections", ["Introduction", "Data Overview", "EDA", 'ML Model','Actual Tomorrow\'s Prediction', "Conclusion"])

if options == "Introduction":
    st.header("Introduction")
    st.write("Bitcoin (BTC) has emerged as a revolutionary digital asset, transforming the global financial landscape with its decentralized nature and blockchain technology. As a highly volatile and widely traded cryptocurrency, analyzing its price trends, market behavior, and technical indicators is crucial for both investors and researchers. This project focuses on visualizing Bitcoin's historical price data, utilizing key indicators such as Simple Moving Average (SMA), Exponential Moving Average (EMA), and Bollinger Bands to uncover meaningful insights. By integrating interactive and insightful visualizations, this analysis aims to provide a comprehensive understanding of BTC's market dynamics. The project's ultimate goal is to assist in identifying trends, volatility patterns, and potential opportunities for informed decision-making in the ever-evolving cryptocurrency market.")
    image_path = os.path.join(os.path.dirname(__file__), "immm.jpg")
    st.image(image_path, caption="BITCOIN", width=670)

if options == "Data Overview":
    st.header("Data Overview")
    st.write("### Dataset Details")
    st.write("The dataset used in this project contains historical Bitcoin price data with the following columns:")
    st.write("1. **open**: The opening price of Bitcoin for a specific day.")
    st.write("2. **high**: The highest price reached during the day.")
    st.write("3. **low**: The lowest price reached during the day.")
    st.write("4. **close**: The closing price of Bitcoin for the day.")
    st.write("5. **volume**: The volume of Bitcoin traded.")
    st.write("6. **SMA_7**: 7-day Simple Moving Average.")
    st.write("7. **EMA_12**: 12-day Exponential Moving Average.")
    st.write("7. **EMA_26**: 26-day Exponential Moving Average.")
    st.write("8. **RSI**: Relative Strength Index, a momentum oscillator.")
    st.write("9. **Bollinger_High**: Upper Bollinger Band.")
    st.write("10. **Bollinger_Low**: Lower Bollinger Band.")
    st.write("### Data Summary")
    
    total_rows = len(btc_data)
    st.write(f"Total number of rows in the dataset:   {total_rows}")

    numerical_columns = ['open', 'high', 'low', 'close', 'SMA_7', 'EMA_12', 'EMA_26', 'RSI', 'Bollinger_High', 'Bollinger_Low']
    summary_stats = btc_data.describe()
    st.write(summary_stats)
    
    st.write("### First Few Rows of the Dataset")
    st.write(btc_data.head(10))

if options == "EDA":
    st.header("Exploratory Data Analysis")
    
    st.write("### Closing Price Over Time")
    plt.figure(figsize=(12, 6))
    plt.plot(btc_data['close'], label='Close Price', color='blue')
    plt.title('Bitcoin Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    st.pyplot(plt)
    st.write("**Explanation:** This graph shows BTC price over time for the last 15 years since its creation.")

    st.write("### Overlay Trend Zones")
    btc_data['Trend'] = np.where(btc_data['close'] > btc_data['EMA_12'], 'Bullish', 'Bearish')
    plt.figure(figsize=(12, 6))
    plt.plot(btc_data['close'], label='Close Price', color='blue')
    plt.plot(btc_data['EMA_12'], label='EMA 12', color='green', linestyle='--')
    plt.plot(btc_data['EMA_26'], label='EMA 26', color='red', linestyle='--')
    plt.fill_between(btc_data.index, btc_data['close'], color='green', where=btc_data['Trend'] == 'Bullish', alpha=0.1, label='Bullish Zone')
    plt.fill_between(btc_data.index, btc_data['close'], color='red', where=btc_data['Trend'] == 'Bearish', alpha=0.1, label='Bearish Zone')
    plt.title('Overlay Trend Zones')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    st.pyplot(plt)
    st.write("**Explanation:** This chart illustrates Bitcoin's price movement with EMAs and trend zones.")

    st.write("### Candlestick Chart with SMA & EMA")
    last_year_data = btc_data.loc[btc_data.index >= (btc_data.index.max() - pd.DateOffset(months=6))]
    mpf_data = last_year_data[['open', 'high', 'low', 'close', 'volume']]
    fig, ax = mpf.plot(
        mpf_data,
        type='candle',
        mav=(7, 12, 26),
        volume=False,
        title='Candlestick Chart with SMA & EMA',
        style='yahoo',
        returnfig=True
    )
    st.pyplot(fig)
    st.write("**Explanation:** Candlestick chart with moving averages for trend analysis.")

    st.write("### Histogram for MACD with Closing Price")
    plt.figure(figsize=(12, 6))
    plt.plot(last_year_data['MACD'], label='MACD Line', color='blue')
    plt.plot(last_year_data['Signal_Line'], label='Signal Line', color='red', linestyle='--')
    plt.bar(last_year_data.index, last_year_data['MACD_Histogram'], label='MACD Histogram', color='green', alpha=0.6)
    plt.title('MACD and Signal Line with Histogram')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    st.pyplot(plt)
    st.write("**Explanation:** MACD indicator showing momentum and trend signals.")

    st.write("### Distribution of Closing Prices")
    plt.figure(figsize=(10, 6))
    sns.histplot(btc_data['close'], bins=50, kde=True, color='blue')
    plt.title('Distribution of Bitcoin Closing Prices')
    plt.xlabel('Price (USD)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    st.write("### Monthly Average Prices")
    btc_data['month'] = btc_data.index.month
    monthly_avg = btc_data.groupby('month')['close'].mean()
    plt.figure(figsize=(10, 6))
    monthly_avg.plot(kind='bar', color='blue')
    plt.title('Average Bitcoin Prices by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Price (USD)')
    st.pyplot(plt)

    st.write("### Correlation Between Features")
    numeric_data = btc_data.select_dtypes(include=['number'])
    corr = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    st.pyplot(plt)

if options == "ML Model":
    st.header("LSTM Model for Bitcoin Price Prediction")

    st.write("### Model Evaluation Metrics")
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "Mean Squared Error": mse,
        "Mean Absolute Error": mae,
        "R-squared Score": r2,
        "Accuracy (%)": accuracy
    }
    st.write(pd.DataFrame([metrics], index=["Metrics"]))

    st.write("### Actual vs Predicted Prices")
    results_df = pd.DataFrame({
        "Date": btc_data.index[-len(y_test):][:10],
        "Actual": y_test.flatten()[:10],
        "Predicted": y_pred.flatten()[:10] 
    })
    st.write(results_df)

    st.write("### Test vs Prediction Plot")
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_test)), y_test, label='Actual', color='blue')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted', color='orange')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Index')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    st.pyplot(plt)

if options == 'Actual Tomorrow\'s Prediction':
    st.header("3-Day Bitcoin Price Prediction")
    
    # Get fresh data
    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=365)  # 1 year history
    new_data = yf.download("BTC-USD", start=start_date, end=end_date)
    new_data = new_data.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume'
    })
    
    # Add indicators
    new_data = add_indicators(new_data)
    
    # Prepare prediction input
    latest_data = new_data[features].tail(window_size)
    scaled_latest = scaler.transform(latest_data)
    sequence = np.expand_dims(scaled_latest, axis=0)
    
    # Make 3-day prediction
    scaled_pred = model.predict(sequence)
    pred_price = scaler.inverse_transform([[0, 0, 0, scaled_pred[0][0], 0, 0, 0]])[0][3]
    pred_date = end_date + dt.timedelta(days=forecast_horizon)
    
    # Display results
    st.write(f"### Predicted Closing Price for {pred_date.strftime('%Y-%m-%d')}:")
    st.write(f"**${pred_price:.2f}**")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(new_data.index, new_data['close'], label='Historical Prices')
    plt.axvline(x=end_date, color='red', linestyle='--', label='Today')
    plt.scatter(pred_date, pred_price, color='green', label=f'{forecast_horizon}-Day Prediction')
    plt.title(f'Bitcoin {forecast_horizon}-Day Price Prediction')
    plt.legend()
    st.pyplot(plt)

if options == "Conclusion":
    st.header("Conclusion")
    st.write("This analysis highlights Bitcoin's price trends and the utility of technical indicators in understanding market behavior. The combination of technical indicators and the LSTM model demonstrated the potential of data-driven approaches in understanding and predicting market behavior.")
    st.write("The ability to forecast Bitcoin prices with accuracy can empower traders and investors to make informed decisions, potentially mitigating risks and maximizing returns in a volatile market.")
    st.write("Future work could explore incorporating external factors, such as market sentiment or macroeconomic indicators, into the model. Additionally, fine-tuning hyperparameters and leveraging advanced machine learning architectures could further enhance prediction accuracy.")

st.markdown(
    """
    <div style="text-align: center; color: #bbbbbb; font-size: 18px; margin-top: 20px;">
        <p>Bitcoin Analysis and Prediction App - Created by Usman Ahmad</p>
    </div>
    """,
    unsafe_allow_html=True
)
