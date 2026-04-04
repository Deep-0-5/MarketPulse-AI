import streamlit as st
import pandas as pd
import time
import requests
from datetime import datetime
from core.ingestor import DataIngester
from core.processor import DataProcessor
from core.engineer import FeatureEngineer
from core.predictor import MarketPredictor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_lottie import st_lottie

# --- PAGE CONFIG ---
st.set_page_config(page_title="MarketPlus AI", layout="wide")

# --- LOTTIE ANIMATION LOADER ---
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load the animation data
lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_gd89vaxz.json")

# --- HEADER WITH ANIMATION CHECK ---
col_t1, col_t2 = st.columns([1, 6])
with col_t1:
    # FIXED: Only try to render if data was successfully loaded
    if lottie_ai:
        st_lottie(lottie_ai, height=80, key="ai_icon")
    else:
        st.write("📈") # Fallback icon if URL fails
with col_t2:
    st.title("MarketPulse AI: 5-Minute Forecaster")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")
ticker_list = ["BTC-USD", "ETH-USD", "AAPL", "TSLA", "MSFT", "GOOGL", "NVDA"]
selected_ticker = st.sidebar.selectbox("Select Popular Ticker", ticker_list)
custom_ticker = st.sidebar.text_input("Or Enter Custom Ticker (e.g. SOL-USD)")

ticker = custom_ticker.upper() if custom_ticker else selected_ticker
refresh_interval = st.sidebar.slider("Refresh Interval (Seconds): ", 10, 300, 60)
auto_refresh = st.sidebar.checkbox("Enable Live Updates", value=True)
timer_placeholder = st.sidebar.empty()

# --- ML PIPELINE INITIALIZATION ---
@st.cache_resource
def initialize_pipeline(ticker_sym):
    bot = DataIngester(ticker=ticker_sym)
    cleaner = DataProcessor()
    engineer = FeatureEngineer()
    predictor = MarketPredictor()
    return bot, cleaner, engineer, predictor

bot, cleaner, engineer, predictor = initialize_pipeline(ticker)

if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = datetime.now()
if 'last_price' not in st.session_state:
    st.session_state.last_price = 0.0

def run_analysis():
    raw_data = bot.fetch_market_data(period="7d", interval="1m")
    if raw_data.empty:
        st.error(f"Data fetch failed for {ticker}.")
        return

    current_price = raw_data['Close'].iloc[-1]
    if current_price != st.session_state.last_price:
        st.session_state.last_update_time = datetime.now()
        st.session_state.last_price = current_price
        # Toast notification for a modern app feel
        st.toast(f"🚀 Price Update for {ticker}: ${current_price:,.2f}")

    # 2. Engineering
    df = cleaner.clean_data(raw_data)
    df = cleaner.add_feature(df)
    df = engineer.add_rsi(df)
    df = engineer.add_trend(df)

    # 3. Prediction
    df_ready = predictor.prepare_target(df)
    accuracy = predictor.train(df_ready)
    
    current_feat = df_ready[predictor.feature_cols].tail(1)
    probabilities = predictor.model.predict_proba(current_feat)[0]
    prediction = predictor.model.predict(current_feat)[0]
    confidence = probabilities[prediction]
    
    # --- SIGNAL & METRICS ---
    st.subheader(f"🔮 AI Analysis: {ticker}")
    if confidence > 0.55:
        if prediction == 1:
            st.success(f"🟢 **STRONG SIGNAL: BUY** ({confidence:.1%} Confidence)")
        else:
            st.warning(f"🟠 **STRONG SIGNAL: WAIT** ({confidence:.1%} Confidence)")
    else:
        st.info(f"⚪ **NO CLEAR SIGNAL** (Confidence: {confidence:.1%})")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${current_price:,.4f}")
    col2.metric("Backtest Accuracy", f"{accuracy:.1%}")
    col3.metric("RSI (14m)", f"{df['RSI'].iloc[-1]:.1f}")
    
    move = current_price * df['Volatility'].iloc[-1] * 2 
    col4.metric("Predicted Target", f"${(current_price + move if prediction == 1 else current_price - move):,.4f}")

    # --- INTERACTIVE TABS ---
    tab1, tab2 = st.tabs(["📊 Interactive Live Chart", "🧠 Feature Statistics (Seaborn)"])
    
    with tab1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.7, 0.3])
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#00ff88')), row=1, col=1)
        
        # Color coding MACD bars for visual impact
        colors = ['#00ff88' if x > 0 else '#ff4b4b' for x in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD', marker_color=colors), row=2, col=1)
        
        fig.update_layout(template='plotly_dark', height=600, margin=dict(l=10, r=10, b=10, t=30), hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write("### Indicator Relationship Analysis")
        # Matplotlib/Seaborn deep-dive for academic/professional appeal
        fig_sns, ax_sns = plt.subplots(figsize=(10, 4))
        plt.style.use("dark_background")
        sns.scatterplot(data=df_ready.tail(500), x='RSI', y='Volatility', hue='Target', palette='viridis', ax=ax_sns)
        ax_sns.set_title("Clustering: RSI vs Volatility (Last 500 Samples)")
        st.pyplot(fig_sns)

    # --- FOOTER ---
    st.markdown("---")
    st.caption("⚠️ **Risk Notice:** Experimental tool for portfolio purposes. Not financial advice.")

# --- EXECUTION ---
run_analysis()

# --- LIVE REFRESH LOOP ---
if auto_refresh:
    for i in range(refresh_interval):
        elapsed = datetime.now() - st.session_state.last_update_time
        timer_placeholder.success(f"✨ Last Price Tick: {elapsed.seconds}s ago")
        time.sleep(1) 
    st.rerun()
else:
    elapsed = datetime.now() - st.session_state.last_update_time
    timer_placeholder.success(f"✨ Last Price Tick: {elapsed.seconds}s ago")