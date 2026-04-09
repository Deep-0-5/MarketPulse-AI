import streamlit as st
import pandas as pd
import time
import requests
from datetime import datetime, timedelta

# Internal Core Imports
from core.ingestor import DataIngester
from core.utils import get_token_map
from core.processor import DataProcessor
from core.engineer import FeatureEngineer
from core.predictor import MarketPredictor

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_lottie import st_lottie

# --- PAGE CONFIG ---
st.set_page_config(page_title="MarketPlus AI", layout="wide", page_icon="📈")

# --- LOTTIE ANIMATION LOADER ---
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_gd89vaxz.json")

# --- HEADER ---
col_t1, col_t2 = st.columns([1, 6])
with col_t1:
    if lottie_ai:
        st_lottie(lottie_ai, height=80, key="ai_icon")
    else:
        st.write("📈")
with col_t2:
    st.title("MarketPulse AI: NSE 5-Minute Forecaster")

# --- SIDEBAR & TOKEN SELECTION ---
st.sidebar.header("Market Selection (NSE)")

# Load the map of 50,000+ tokens from Angel One
try:
    token_dict = get_token_map()
    search_query = st.sidebar.text_input("Search Stock (e.g., SBIN, RELIANCE)", value="SBIN")
    
    # Filter tokens based on search
    filtered_stocks = [s for s in token_dict.keys() if search_query.upper() in s]
    
    if filtered_stocks:
        selected_stock = st.sidebar.selectbox("Select Result", filtered_stocks)
        token = token_dict[selected_stock]
        st.sidebar.success(f"Token: {token} Selected")
    else:
        st.sidebar.warning("No stock found.")
        token = None
except Exception as e:
    st.sidebar.error("Could not load tokens. Check internet connection.")
    token = None

st.sidebar.markdown("---")
refresh_interval = st.sidebar.slider("Refresh Interval (Sec):", 30, 300, 60)
auto_refresh = st.sidebar.checkbox("Enable Live Updates", value=True)
timer_placeholder = st.sidebar.empty()

# --- ML PIPELINE INITIALIZATION ---
@st.cache_resource
def initialize_pipeline():
    # Initialize components
    bot = DataIngester() 
    cleaner = DataProcessor()
    engineer = FeatureEngineer()
    predictor = MarketPredictor()
    return bot, cleaner, engineer, predictor

bot, cleaner, engineer, predictor = initialize_pipeline()

# Session State Management
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'last_price' not in st.session_state:
    st.session_state.last_price = 0.0

# --- CORE ANALYSIS FUNCTION ---
def run_analysis():
    if not token:
        st.info("Please select a stock from the sidebar to begin analysis.")
        return

    # 1. Ingestion (Angel One API)
    with st.spinner(f"Fetching live data for {selected_stock}..."):
        raw_data = bot.fetch_data(token=token, interval="ONE_MINUTE")
    
    if raw_data.empty:
        st.error("Data fetch failed. Ensure your API keys and TOTP are correct in Secrets.")
        return

    # Align Angel One columns to our Processor
    # Angel returns: ['Time', 'Open', 'High', 'Low', 'Close', 'Vol']
    raw_data = raw_data.rename(columns={'Time': 'Timestamp', 'Vol': 'Volume'})
    
    # 2. Processing & Engineering
    df = cleaner.clean_data(raw_data)
    df = cleaner.add_feature(df)
    df = engineer.add_rsi(df)
    df = engineer.add_trend(df)

    # 3. ML Prediction
    df_ready = predictor.prepare_target(df)
    accuracy = predictor.train(df_ready) # Dynamic training on recent history
    
    current_feat = df_ready[predictor.feature_cols].tail(1)
    probabilities = predictor.model.predict_proba(current_feat)[0]
    prediction = predictor.model.predict(current_feat)[0]
    confidence = probabilities[prediction]
    
    current_price = df['Close'].iloc[-1]
    
    # --- UI RENDERING ---
    st.subheader(f"🔮 AI Signal: {selected_stock}")
    
    if confidence > 0.52:
        if prediction == 1:
            st.success(f"🟢 **BUY SIGNAL** ({confidence:.1%} Confidence)")
        else:
            st.warning(f"🟠 **WAIT/SELL SIGNAL** ({confidence:.1%} Confidence)")
    else:
        st.info(f"⚪ **NEUTRAL** (Confidence: {confidence:.1%})")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Live Price", f"₹{current_price:,.2f}")
    m2.metric("ML Accuracy", f"{accuracy:.1%}")
    m3.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    
    target_move = current_price * df['Volatility'].iloc[-1]
    predicted_val = current_price + target_move if prediction == 1 else current_price - target_move
    m4.metric("5m Target", f"₹{predicted_val:,.2f}")

    # --- TABS ---
    t1, t2 = st.tabs(["📊 Technical Chart", "🧠 Feature Deep-Dive"])
    
    with t1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#00ff88', width=2)), row=1, col=1)
        
        # MACD Visualization
        colors = ['#00ff88' if x > 0 else '#ff4b4b' for x in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Momentum', marker_color=colors), row=2, col=1)
        
        fig.update_layout(template='plotly_dark', height=500, margin=dict(l=10, r=10, b=10, t=10), hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.write("### Statistical Correlation (RSI vs Volatility)")
        fig_sns, ax_sns = plt.subplots(figsize=(10, 4))
        plt.style.use("dark_background")
        sns.scatterplot(data=df_ready.tail(300), x='RSI', y='Volatility', hue='Target', palette='magma', ax=ax_sns)
        st.pyplot(fig_sns)

# --- EXECUTION ---
run_analysis()

# --- LIVE REFRESH ---
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()