import pandas as pd

class FeatureEngineer:
    """
    module 3
    work: create technical indicator to give AI context
    """
    def add_rsi(self,df,window=14):
        delta=df['Close'].diff()
        gain=(delta.where(delta>0,0)).rolling(window=window).mean()
        loss=(-delta.where(delta<0,0)).rolling(window=window).mean()
        
        rs=gain/loss
        
        df['RSI']=100-(100/(1+rs))
        return df
    
    def add_trend(self,df,window=20):
        """
        calculate average price of past 20 min
        """
        df['SMA_20']=df['Close'].rolling(window=window).mean()
        
        # Distance from SMA: Tells us if the price is 'stretched' far from the average
        df['Price_Dist_SMA'] = df['Close'] / df['SMA_20']
        
        return df.dropna()