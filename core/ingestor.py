from SmartApi import SmartConnect
import pyotp
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

class DataIngester:
    def __init__(self):
        self.api_key = st.secrets["angel_one"]["api_key"]
        self.client_id = st.secrets["angel_one"]["client_id"]
        self.password = st.secrets["angel_one"]["password"]
        self.totp_token = st.secrets["angel_one"]["totp_token"]
        
        self.smartApi = SmartConnect(api_key=self.api_key)
        self.login()

    def login(self):
        try:
            totp = pyotp.TOTP(self.totp_token).now()
            self.smartApi.generateSession(self.client_id, self.password, totp)
        except Exception as e:
            st.error(f"Login Failed: {e}")

    def fetch_data(self, token, interval="ONE_MINUTE"):
        # Fetching last 2 days of 1-minute data
        to_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        from_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d %H:%M')
        
        params = {
            "exchange": "NSE",
            "symboltoken": str(token),
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date
        }
        
        res = self.smartApi.getCandleData(params)
        if res.get('status') and res.get('data'):
            df = pd.DataFrame(res['data'], columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)
            return df
        return pd.DataFrame()