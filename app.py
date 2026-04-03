import streamlit as st
import pandas as pd
import time
from datetime import datetime
from core.ingestor import DataIngester
from core.processor import DataProcessor
from core.engineer import FeatureEngineer
from core.predictor import MarketPredictor
import plotly.graph_objects as go

st.set_page_config(page_title="MarketPlus AI", layout="wide")
st.title("📈 MarketPulse AI: 5-Minute Forecaster")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter Ticker", value="BTC-USD")

# Auto-refresh set to 60 seconds (1 minute interval)
referesh_interval=st.sidebar.slider("Referesh Intervals(In Seconds): ",5,60,10)
auto_refresh = st.sidebar.checkbox("Enable 1-Minute Live Updates", value=True)

if auto_refresh:
    countdown_placeholder=st.sidebar.empty()

# --- PIPELINE ---
@st.cache_resource
def initialize_pipeline(ticker_sym):
    bot = DataIngester(ticker=ticker_sym)
    cleaner = DataProcessor()
    engineer = FeatureEngineer()
    predictor = MarketPredictor()
    return bot, cleaner, engineer, predictor

bot,cleaner,engineer,predictor=initialize_pipeline(ticker)

def run_analysis():
    # 1. Fetch 1-minute data candles
    raw_data = bot.fetch_market_data(period="2d", interval="1m")
    if raw_data.empty:
        st.error("Data fetch failed.")
        return

    # 2. Process and Feature Engineering
    df = cleaner.clean_data(raw_data)
    df = cleaner.add_feature(df)
    df = engineer.add_rsi(df)
    df = engineer.add_trend(df)

    # 3. Predict 5 Minutes Ahead
    df_ready = predictor.prepare_target(df)
    accuracy = predictor.train(df_ready)
    
    current_feat = df_ready[['Log_Returns','Volatility','RSI','Price_Dist_SMA']].tail(1)
    prediction = predictor.model.predict(current_feat)[0]
    
    # --- UI: 5-MINUTE PREDICTION ---
    st.subheader("🔮 5-Minute Prediction Signal")
    if prediction == 1:
        st.success(f"🟢 **SIGNAL: BUY** - AI predicts {ticker} will be HIGHER in 5 minutes.")
    else:
        st.warning(f"🟠 **SIGNAL: WAIT** - AI predicts {ticker} will be LOWER or FLAT in 5 minutes.")

    # --- UI: METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    curr_p = df["Close"].iloc[-1]
    
    # Precision fix for low-value tickers
    col1.metric("Current Price", f"${curr_p:,.4f}")
    col2.metric("Prediction Accuracy", f"{accuracy:.1%}")
    col3.metric("RSI (14m)", f"{df['RSI'].iloc[-1]:.1f}")
    
    # Target Price based on Volatility for the 5-min window
    move = curr_p * df['Volatility'].iloc[-1] * 2 # multiplier for 5-min expected move
    targ_p = curr_p + move if prediction == 1 else curr_p - move
    
    col4.metric("Predicted Price (5m)", f"${targ_p:,.4f}", delta=f"{move:,.4f}")

    # --- CHART ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#00ff88')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(dash='dot')))
    fig.update_layout(template='plotly_dark', height=400)
    st.plotly_chart(fig, use_container_width=True)

# Run once
run_analysis()

# Auto-Refresh Logic
if auto_refresh:
    for i in range(referesh_interval,0,-1):
        countdown_placeholder.write(f"🔄 Next update in: {i} seconds")
        time.sleep(1) # Wait 1 minute
    st.rerun()