import yfinance as yf
import pandas as pd

class DataIngester:
    """
    Module 1: The ingester
    work: connect to live api and get market ticks    
    """
    def __init__(self,ticker="BTC-USD"):
        self.ticker=ticker
        
    def fetch_market_data(self, period="1d",interval="1m"):
        print(f"🚀 MarketPulse AI: Fetching live data for {self.ticker}...")
        try:
            df=yf.download(self.ticker,period=period,interval=interval)
            
            if isinstance(df.columns,pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            return df
        
        except Exception as e:
            print(f"❌ Error: Could not connect to API. {e}")
            return pd.DataFrame()

