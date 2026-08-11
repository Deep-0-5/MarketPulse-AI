# MarketPulse AI

MarketPulse AI is a Streamlit dashboard for educational NSE market analysis. It fetches recent intraday candle data, builds technical features, and uses a Random Forest model to classify short-term market direction as a bullish, bearish/caution, or neutral educational signal.

> Educational use only: MarketPulse AI is not financial advice, investment advice, or a guarantee of future returns.

## Current Features

- NSE stock search using Angel One symbol tokens
- Live intraday candle ingestion through Angel One SmartAPI
- Technical indicators including RSI, volatility, Bollinger Band position, and MACD
- Random Forest based 5-minute educational market signal
- Plotly technical chart with momentum visualization
- Feature deep-dive chart for RSI and volatility
- Signal explanation panel with directional bias, confidence, and projection context
- Free Preview vs Pro Beta positioning inside the dashboard

## Phase 1 Safety Checklist

- Rotate any Angel One credentials that were used during local development before deploying publicly.
- Keep real credentials only in `.streamlit/secrets.toml` locally or in your hosting provider's secrets manager.
- Do not commit `.streamlit/secrets.toml`, `.env`, broker credentials, API keys, passwords, or TOTP tokens.
- Use safer product wording: educational signal, bullish/bearish/neutral indicator, confidence score, and projection.
- Avoid promising profit, guaranteed calls, or personalized investment advice.

## Local Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create local secrets:

   ```bash
   cp .streamlit/secrets.example.toml .streamlit/secrets.toml
   ```

3. Add your fresh Angel One credentials to `.streamlit/secrets.toml`.

4. Configure temporary local access in `.streamlit/secrets.toml`:

   ```toml
   [access]
   mode = "mock"
   demo_email = "demo@marketpulse.ai"
   demo_password = "marketpulse-demo"
   payment_url = "https://rzp.io/l/your-marketpulse-trial-link"
   razorpay_key_id = "your_razorpay_key_id"
   razorpay_key_secret = "your_razorpay_key_secret"
   razorpay_plan_id = "your_razorpay_plan_id"
   ```

5. Run the app:

   ```bash
   streamlit run app.py
   ```

## Commercial Launch Notes

For a paid beta, position this as an educational AI market dashboard rather than a trading-call service. Before selling access, add login/access control, trial handling, and payment verification through the chosen membership platform.

## Product Packaging

### Free Preview

- Stock search
- Live educational signal
- Technical chart
- Basic indicator notes

### Pro Beta

- 3-day trial, then paid membership
- Saved watchlists
- Daily market summary
- Weekly educational report
- Login and paid access gating through Razorpay subscriptions

## Next Build Phase

The next build step should connect Razorpay subscription verification before public launch. The first practical production version can show a locked dashboard until the user has an active trial or paid subscription.

## Phase 3 Temporary Access Gate

The app currently uses a local demo email/password so the paid-user experience can be tested before Razorpay verification is connected.

- Users enter an email and password before the dashboard loads.
- Valid demo credentials unlock the dashboard for the current Streamlit session.
- The sidebar shows the logged-in email and a logout button.
- Invalid access shows a message and the Razorpay trial/payment link.
- Broker token loading and Angel One login happen only after access is granted.
- Live market updates are off by default to reduce broker API rate-limit errors.
- Market data is cached for 90 seconds, and users can refresh manually from the sidebar.

This is not production authentication. Replace the demo credential check with Razorpay subscription verification or a webhook-backed active-user database before public launch.

## Razorpay Integration Plan

- Create a Razorpay plan for the monthly Pro Beta price.
- Create a subscription flow with a 3-day trial period.
- Add the Razorpay payment or subscription link as `[access].payment_url`.
- Use webhooks to record active, trialing, cancelled, or expired subscriptions.
- Replace the local demo key check with a server-side lookup against the active subscription records.
