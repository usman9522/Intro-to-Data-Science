import numpy as np # linear algebra
import streamlit as st
import pandas as pd 
import seaborn as sns
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime as dt
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load dataset
btc_data = pd.read_csv("BTC.csv")
btc_data['date'] = pd.to_datetime(btc_data['date'])
btc_data.set_index('date', inplace=True)

# Add technical indicators
btc_data['SMA_7'] = SMAIndicator(btc_data['close'], window=7).sma_indicator()
btc_data['EMA_12'] = EMAIndicator(btc_data['close'], window=12).ema_indicator()
btc_data['EMA_26'] = EMAIndicator(btc_data['close'], window=26).ema_indicator()
btc_data['RSI'] = RSIIndicator(btc_data['close']).rsi()
bbands = BollingerBands(btc_data['close'])
btc_data['Bollinger_High'] = bbands.bollinger_hband()
btc_data['Bollinger_Low'] = bbands.bollinger_lband()

# Recalculate MACD and Signal Line
btc_data['MACD'] = btc_data['EMA_12'] - btc_data['EMA_26']
btc_data['Signal_Line'] = btc_data['MACD'].rolling(window=9).mean()
btc_data['MACD_Histogram'] = btc_data['MACD'] - btc_data['Signal_Line']

# Drop rows with NaN values due to indicator calculations
btc_data = btc_data.dropna()

## MACHINE LEArNING
features = ['close', 'SMA_7', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'Signal_Line']
target = 'close'
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(btc_data[features])
scaled_target = scaler.fit_transform(btc_data[[target]])

X, y = [], []
window_size = 30  # Use the past 30 days for predictions
for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i])
    y.append(scaled_target[i])
X, y = np.array(X), np.array(y)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
if "model" not in st.session_state:
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

# Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(f"X_train shape: {X_train.shape}")
    print(f"Model input shape: {model.input_shape}")


# Train the model
    with st.spinner("Training the model. Please wait..."):
        model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1)
        
        st.session_state.model = model
        st.success("Model trained and ready!")
else:
    # Load the model from session state
    model = st.session_state.model
    #st.info("Model loaded from session state.")

# Make predictions
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Calculate accuracy percentage
accuracy = 100 - np.mean(np.abs((y_test - y_pred) / y_test)) * 100



# Streamlit App
st.title("Bitcoin Price Analysis and Insights")

st.sidebar.header("Sections")
options = st.sidebar.radio("Sections", ["Introduction", "Data Overview", "EDA", 'ML Model','Actual Tomorrow\'s Prediction', "Conclusion"])

if options == "Introduction":
    st.header("Introduction")
    st.write("Bitcoin (BTC) has emerged as a revolutionary digital asset, transforming the global financial landscape with its decentralized nature and blockchain technology. As a highly volatile and widely traded cryptocurrency, analyzing its price trends, market behavior, and technical indicators is crucial for both investors and researchers. This project focuses on visualizing Bitcoin's historical price data, utilizing key indicators such as Simple Moving Average (SMA), Exponential Moving Average (EMA), and Bollinger Bands to uncover meaningful insights. By integrating interactive and insightful visualizations, this analysis aims to provide a comprehensive understanding of BTC's market dynamics. The project's ultimate goal is to assist in identifying trends, volatility patterns, and potential opportunities for informed decision-making in the ever-evolving cryptocurrency market.")
   # st.write("This project aims to analyze Bitcoin's historical data and provide key insights into its price trends using statistical and technical tools.")
   # st.write("Various technical indicators like SMA, EMA, RSI, and Bollinger Bands are incorporated to understand market behavior.")
    st.image("immm.jpg", caption="BITCOIN", width=670)

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

 
    numerical_columns = ['open', 'high', 'low', 'close' , 'SMA_7', 'EMA_12' , 'EMA_26' , 'RSI' , 'Bollinger_High' , 'bollinger_Low' ]
    summary_stats = btc_data.describe()
    #summary_stats.columns = ['Min', 'Max', 'Average', 'Count']
    st.write(summary_stats)
    
    st.write("### First Few Rows of the Dataset")
    st.write(btc_data.head(10))
    
    

if options == "EDA":
    st.header("Exploratory Data Analysis")
      # Visual 2: Overlay Trend Zones
      
      
    
    # Visual 1: Closing Price Over Time
    st.write("### Closing Price Over Time")
    plt.figure(figsize=(12, 6))
    plt.plot(btc_data['close'], label='Close Price', color='blue')
    plt.title('Bitcoin Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    st.pyplot(plt)
    st.write("**Explanation:**")
    st.write("This graph is simply representing btc price over time for the last 15 years since its creation. ")
    st.write()


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
    st.write("**Explanation:**")
    st.write("This chart illustrates Bitcoin's price movement over time, with overlays of the 12-day EMA and 26-day EMA to highlight trends. The chart also includes shaded regions to differentiate bullish and bearish zones, providing a clear visual representation of market trends and momentum shifts.")
    st.write()
    # Assuming 'btc_data' is your DataFrame with the necessary columns
    # Ensure columns: 'date', 'open', 'high', 'low', 'close'

    st.write("### Candlestick Chart with SMA & EMA")

    # Filter last year's data
    last_year_data = btc_data.loc[btc_data.index >= (btc_data.index.max() - pd.DateOffset(months=6))]
    mpf_data = last_year_data.reset_index()
    mpf_data.set_index('date', inplace=True)

    # Create mplfinance candlestick chart as a Matplotlib Figure
    fig, ax = mpf.plot(
        mpf_data,
        type='candle',
        mav=(7, 12, 26),
        volume=False,
        title='Candlestick Chart with SMA & EMA',
        style='yahoo',
        returnfig=True  # Return the figure object
    )

    # Display the plot in Streamlit
    st.pyplot(fig)
    st.write("**Explanation:**")
    st.write("This candlestick chart visualizes Bitcoin's price movements over the last six months, overlaid with the Simple Moving Average (SMA) and Exponential Moving Averages (EMA) for trend analysis. The SMA and EMA lines provide insight into short- and long-term price trends, helping to identify potential buy and sell signals.This candlestick chart visualizes Bitcoin's price movements over a specific period, overlaid with the Simple Moving Average (SMA) and Exponential Moving Averages (EMA) for trend analysis. The SMA and EMA lines provide insight into short- and long-term price trends, helping to identify potential buy and sell signals.")
    st.write()
    
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
    st.write("**Explanation:**")
    st.write("This chart illustrates the MACD (Moving Average Convergence Divergence) indicator with its components: the MACD Line, Signal Line, and Histogram. The MACD Line represents momentum by showing the difference between two EMAs, while the Signal Line acts as a smoother trend indicator. The histogram highlights the divergence between the MACD and Signal Lines, aiding in identifying potential buy and sell signals based on crossovers and the momentum of price changes.")
    st.write()


    # Visual 2: Distribution of Closing Prices
    st.write("### Distribution of Closing Prices")
    plt.figure(figsize=(10, 6))
    sns.histplot(btc_data['close'], bins=50, kde=True, color='blue')
    plt.title('Distribution of Bitcoin Closing Prices')
    plt.xlabel('Price (USD)')
    plt.ylabel('Frequency')
    st.pyplot(plt)
    st.write("**Explanation:**")
    st.write("This histogram shows the distribution of Bitcoin's closing prices, with most values concentrated in the lower range and fewer instances at higher prices, reflecting its historical volatility. ")
    st.write()

    # Visual 3: Monthly Average Prices
    st.write("### Monthly Average Prices")
    btc_data['month'] = btc_data.index.month
    monthly_avg = btc_data.groupby('month')['close'].mean()
    plt.figure(figsize=(10, 6))
    monthly_avg.plot(kind='bar', color='blue')
    plt.title('Average Bitcoin Prices by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Price (USD)')
    st.pyplot(plt)
    st.write("**Explanation:**")
    st.write("This bar chart displays the average Bitcoin prices by month, highlighting seasonal trends and variations in price levels throughout the year. ")
    st.write()

    # Visual 4: Correlation Heatmap
    st.write("### Correlation Between Features")
    numeric_data = btc_data.select_dtypes(include=['number'])
    corr = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    st.pyplot(plt)
    st.write("**Explanation:**")
    st.write("This represents the linear relationship between features. ")
    st.write()
    
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

    # Create a DataFrame with dates, actual, and predicted prices
    results_df = pd.DataFrame({
        "Date": btc_data.index[-len(y_test):][:10],  # Get the last `len(y_test)` dates, limit to 10 rows
        "Actual": y_test.flatten()[:10],
        "Predicted": y_pred.flatten()[:10] 
    })

    # Display the DataFrame
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
      
    ###########################################
    # 1) Download BTC Data from Yahoo Finance
    ###########################################
    # Define the start and end date for data extraction
    start_date = "2017-01-01"
    end_date = dt.datetime.today().strftime('%Y-%m-%d')

    # Download BTC-USD data
    df = yf.download("BTC-USD", start=start_date, end=end_date, interval='1d')
    df.dropna(inplace=True)

    # Clean and rename columns
    df.reset_index(inplace=True)
    df.rename(columns={
        'Date': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)

    # Re-order and sort
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    ###########################################
    # 2) Add Technical Indicators
    ###########################################
    # Compute some sample technical indicators (SMA, RSI, MACD)
    import ta
    df['sma_14'] = ta.trend.SMAIndicator(close=df['close'], window=14).sma_indicator()
    df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # Drop rows with NaN values resulting from indicator calculations
    df.dropna(inplace=True)

    ###########################################
    # 3) Prepare the Data for LSTM
    ###########################################
    # Scale the features
    scaler = MinMaxScaler()
    # Ensure the feature set matches the model's input during training
    scaled_data = scaler.fit_transform(df[['close', 'sma_14', 'ema_12', 'ema_26', 'rsi_14', 'macd', 'macd_signal']])

# Create sequences for LSTM
    sequence_length = 30  # Use the past 30 days to predict the next day
    X = []
    y = []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])  # Past 30 days
        y.append(scaled_data[i, 3])  # 'close' price as the target

    X = np.array(X)
    y = np.array(y)

# Train-test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

###########################################
# Update the last_sequence for prediction
###########################################
    last_sequence = scaled_data[-sequence_length:]  # Last 60 days
    last_sequence = np.expand_dims(last_sequence, axis=0)  # Add batch dimension

# Validate the shape of last_sequence
    assert last_sequence.shape == (1, sequence_length, X_train.shape[2]), \
        f"Shape mismatch: last_sequence {last_sequence.shape}, expected (1, {sequence_length}, {X_train.shape[2]})"

# Predict the next day's scaled price
    next_day_scaled = model.predict(last_sequence)
    next_day_price = scaler.inverse_transform([[0, 0, 0, next_day_scaled[0][0], 0, 0, 0]])[0][3]


    # Determine the next day's date
    last_date = df['timestamp'].max()
    next_date = last_date + pd.Timedelta(days=1)

    # Compare with today's price
    today_price = df.iloc[-1]['close']
    price_change = "higher" if next_day_price > today_price else "lower"
    price_change_color = "green" if next_day_price > today_price else "red"

    st.write(f"### Predicted Closing Price for {next_date.strftime('%Y-%m-%d')}:")
    st.write(f"**${next_day_price:.2f}**")
    st.markdown(f"<span style='color:{price_change_color};font-size:18px'>The predicted price is {price_change} than today's closing price (${today_price:.2f}).</span>", unsafe_allow_html=True)

    ###########################################
    # 6) Visualization
    ###########################################
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['close'], label='Historical Close Prices', color='blue')
    plt.axvline(x=last_date, color='black', linestyle='--', label='Last Known Date')
    plt.scatter(next_date, next_day_price, color=price_change_color, label='Next Day Prediction')
    plt.xlabel('Date')
    plt.ylabel('BTC Price (USD)')
    plt.title('Next Day Price Prediction with LSTM')
    plt.legend()
    st.pyplot(plt)

        


if options == "Conclusion":
    st.header("Conclusion")
    st.write("This analysis highlights Bitcoin's price trends and the utility of technical indicators in understanding market behavior.This project provided an in-depth analysis of Bitcoin price trends, leveraging statistical tools and machine learning techniques. The combination of technical indicators and the LSTM model demonstrated the potential of data-driven approaches in understanding and predicting market behavior.")
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