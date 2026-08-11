import streamlit as st
import pandas as pd
import time
import requests
import hmac
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
st.set_page_config(page_title="MarketPulse AI", layout="wide", page_icon=":chart_with_upwards_trend:")

ACCENT = "#16a34a"
WARNING = "#f59e0b"
DANGER = "#ef4444"
MUTED = "#64748b"

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetric"] {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] {
            color: #cbd5e1;
        }
        .mp-hero {
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 18px 20px;
            background: linear-gradient(135deg, #0f172a 0%, #111827 55%, #0b1220 100%);
            margin-bottom: 18px;
        }
        .mp-eyebrow {
            color: #38bdf8;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .mp-hero h1 {
            margin: 0;
            font-size: 2.2rem;
            line-height: 1.12;
        }
        .mp-hero p {
            color: #cbd5e1;
            margin: 8px 0 0;
            max-width: 860px;
        }
        .mp-note {
            color: #94a3b8;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_signal_profile(prediction, confidence):
    if confidence <= 0.52:
        return {
            "label": "Neutral Educational Signal",
            "tone": "info",
            "direction": "Sideways / unclear",
            "summary": "The model does not have enough confidence to mark a strong directional setup.",
        }

    if prediction == 1:
        return {
            "label": "Bullish Educational Signal",
            "tone": "success",
            "direction": "Upward bias",
            "summary": "Recent price action and indicators are leaning positive in the short term.",
        }

    return {
        "label": "Bearish/Caution Educational Signal",
        "tone": "warning",
        "direction": "Downward or caution bias",
        "summary": "Recent price action and indicators are showing short-term weakness or caution.",
    }


def explain_indicator(name, value):
    if name == "RSI":
        if value >= 70:
            return "RSI is elevated, so the stock may be stretched in the short term."
        if value <= 30:
            return "RSI is low, so the stock may be oversold in the short term."
        return "RSI is in a balanced zone, so momentum is not at an extreme."

    if name == "MACD":
        return "MACD histogram shows whether recent momentum is strengthening or weakening."

    return "Volatility measures how much the stock has been moving recently."


def get_access_settings():
    try:
        access = st.secrets["access"]
    except Exception:
        access = {}

    return {
        "mode": str(access.get("mode", "mock")).lower(),
        "demo_email": str(access.get("demo_email", "demo@marketpulse.ai")).strip().lower(),
        "demo_password": str(access.get("demo_password", "marketpulse-demo")),
        "payment_url": str(access.get("payment_url", "https://razorpay.com/")),
        "razorpay_key_id": str(access.get("razorpay_key_id", "")),
        "razorpay_key_secret": str(access.get("razorpay_key_secret", "")),
        "razorpay_plan_id": str(access.get("razorpay_plan_id", "")),
    }


def initialize_access_state():
    if "has_access" not in st.session_state:
        st.session_state.has_access = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""


def verify_mock_access(email, password, settings):
    if not email.strip() or "@" not in email:
        return False, "Enter your login email."
    if email.strip().lower() != settings["demo_email"]:
        return False, "Invalid email or password."
    if not hmac.compare_digest(password, settings["demo_password"]):
        return False, "Invalid email or password."
    return True, "Access granted."


def verify_razorpay_access(email, password, settings):
    if not email.strip() or "@" not in email:
        return False, "Enter the email you used for your Razorpay trial or subscription."
    if not password.strip():
        return False, "Enter your password."
    if not settings["razorpay_key_id"] or not settings["razorpay_key_secret"] or not settings["razorpay_plan_id"]:
        return (
            False,
            "Razorpay mode is selected, but Razorpay API settings are not configured yet. "
            "Keep access mode as mock until the Razorpay plan and API keys are ready.",
        )

    return (
        False,
        "Razorpay verification is prepared but not connected yet. "
        "The next step is wiring this function to Razorpay subscription verification or webhooks.",
    )


def verify_access(email, password, settings):
    if settings["mode"] == "mock":
        return verify_mock_access(email, password, settings)
    if settings["mode"] == "razorpay":
        return verify_razorpay_access(email, password, settings)
    return False, "Invalid access mode. Use `mock` or `razorpay` in Streamlit secrets."


def render_access_gate():
    initialize_access_state()
    if st.session_state.has_access:
        return

    settings = get_access_settings()
    st.markdown("### MarketPulse AI Pro Access")
    st.write(
        "Log in with your email and password to unlock the dashboard. "
        "Mock mode works locally today; Razorpay mode will verify active subscriptions next."
    )
    st.caption(f"Access mode: {settings['mode']}")

    form_col, info_col = st.columns([1, 1])
    with form_col:
        with st.form("access_gate"):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            has_access, message = verify_access(email, password, settings)
            if has_access:
                st.session_state.has_access = True
                st.session_state.user_email = email.strip()
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    with info_col:
        st.markdown("#### Pro Beta Includes")
        st.write("- 3-day trial access")
        st.write("- Live educational NSE dashboard")
        st.write("- Signal confidence and indicator explanations")
        st.write("- Planned watchlists and daily summaries")
        st.markdown(f"[Start 3-Day Trial]({settings['payment_url']})")

    if settings["mode"] == "mock":
        st.info("Local testing mode: set `[access].demo_email` and `[access].demo_password` in Streamlit secrets.")
    else:
        st.info(
            "Razorpay mode: configure `[access].razorpay_key_id`, "
            "`[access].razorpay_key_secret`, and `[access].razorpay_plan_id` before launch."
        )
    st.stop()

# --- LOTTIE ANIMATION LOADER ---
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_gd89vaxz.json")

# --- HEADER ---
hero_left, hero_right = st.columns([5, 1])
with hero_left:
    st.markdown(
        """
        <div class="mp-hero">
            <div class="mp-eyebrow">NSE education dashboard</div>
            <h1>MarketPulse AI</h1>
            <p>
                5-minute market signals, technical context, and model confidence for learning
                how short-term NSE setups behave.
            </p>
            <p class="mp-note">
                Educational market analysis only. Not financial advice, investment advice,
                or a guarantee of future returns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    if lottie_ai:
        st_lottie(lottie_ai, height=120, key="ai_icon")
    else:
        st.write("MarketPulse")

render_access_gate()

# --- SIDEBAR & TOKEN SELECTION ---
st.sidebar.header("MarketPulse AI")
st.sidebar.caption("Free preview now. Pro access and Razorpay trial gating come next.")
st.sidebar.success(f"Logged in as {st.session_state.user_email}")
if st.sidebar.button("Logout", use_container_width=True):
    st.session_state.has_access = False
    st.session_state.user_email = ""
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.markdown("### Market Selection")

# Load the map of 50,000+ tokens from Angel One
try:
    token_dict = get_token_map()
    search_query = st.sidebar.text_input("Search Stock (e.g., SBIN, RELIANCE)", value="SBIN")
    
    # Filter tokens based on search
    filtered_stocks = [s for s in token_dict.keys() if search_query.upper() in s]
    
    if filtered_stocks:
        selected_stock = st.sidebar.selectbox("Select Result", filtered_stocks)
        token = token_dict[selected_stock]
        st.sidebar.success(f"{selected_stock} selected")
    else:
        st.sidebar.warning("No stock found.")
        token = None
except Exception as e:
    st.sidebar.error("Could not load tokens. Check internet connection.")
    token = None

st.sidebar.markdown("---")
st.sidebar.info(
    "Educational tool only. Signals are model-based market indicators, not buy/sell advice."
)
refresh_interval = st.sidebar.slider("Refresh Interval (Sec):", 120, 600, 180)
auto_refresh = st.sidebar.checkbox("Enable Live Updates", value=False)
if st.sidebar.button("Refresh Market Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
timer_placeholder = st.sidebar.empty()
st.sidebar.markdown("---")
st.sidebar.markdown("### Pro Beta")
st.sidebar.write("Planned: login, 3-day trial, watchlists, and daily summaries.")

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


@st.cache_data(ttl=90, show_spinner=False)
def fetch_market_data(_bot, token, interval):
    return _bot.fetch_data(token=token, interval=interval)

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
        try:
            raw_data = fetch_market_data(bot, token=token, interval="ONE_MINUTE")
        except Exception as e:
            error_text = str(e)
            if "exceeding access rate" in error_text.lower():
                st.error("Angel One rate limit reached. Please wait 10-15 minutes before refreshing again.")
                st.info("Live updates are off by default now, and market data is cached for 90 seconds.")
            else:
                st.error(f"Data fetch failed: {error_text}")
            return
    
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
    signal = get_signal_profile(prediction, confidence)
    latest_rsi = df['RSI'].iloc[-1]
    latest_volatility = df['Volatility'].iloc[-1]
    latest_macd_hist = df['MACD_Hist'].iloc[-1]

    st.subheader(f"{selected_stock} Market Snapshot")
    
    if signal["tone"] == "success":
        st.success(f"**{signal['label']}** ({confidence:.1%} confidence)")
    elif signal["tone"] == "warning":
        st.warning(f"**{signal['label']}** ({confidence:.1%} confidence)")
    else:
        st.info(f"**{signal['label']}** ({confidence:.1%} confidence)")

    st.caption(
        "Signals are generated from recent price action and technical indicators. "
        "Use them for learning and research only."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Live Price", f"INR {current_price:,.2f}")
    m2.metric("Backtest Accuracy", f"{accuracy:.1%}")
    m3.metric("RSI", f"{latest_rsi:.1f}")
    
    target_move = current_price * latest_volatility
    predicted_val = current_price + target_move if prediction == 1 else current_price - target_move
    m4.metric("5m Educational Projection", f"INR {predicted_val:,.2f}")

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown("#### Signal Explanation")
        st.write(signal["summary"])
        st.write(f"**Directional bias:** {signal['direction']}")
        st.write(f"**Confidence:** {confidence:.1%}")
        st.write(f"**Projection range:** +/- INR {target_move:,.2f} from the latest price")
    with c2:
        st.markdown("#### Indicator Notes")
        st.write(f"**RSI:** {explain_indicator('RSI', latest_rsi)}")
        st.write(f"**MACD:** {explain_indicator('MACD', latest_macd_hist)}")
        st.write(f"**Volatility:** {explain_indicator('Volatility', latest_volatility)}")

    # --- TABS ---
    t1, t2, t3 = st.tabs(["Technical Chart", "Feature Deep-Dive", "Free vs Pro"])
    
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

    with t3:
        free_col, pro_col = st.columns(2)
        with free_col:
            st.markdown("#### Free Preview")
            st.write("- Stock search")
            st.write("- Live educational signal")
            st.write("- Technical chart")
            st.write("- Basic indicator notes")
        with pro_col:
            st.markdown("#### Pro Beta")
            st.write("- 3-day trial, then paid membership")
            st.write("- Saved watchlists")
            st.write("- Daily market summary")
            st.write("- Weekly educational report")

# --- EXECUTION ---
run_analysis()

# --- LIVE REFRESH ---
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
