import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    module 3
    work: create technical indicator to give AI context
    """
    def add_rsi(self,df,window=14):
        delta=df['Close'].diff()
        gain=(delta.where(delta>0,0)).rolling(window=window).mean()
        loss=(-delta.where(delta<0,0)).rolling(window=window).mean()
        
        rs=gain/(loss+1e-9)
        df['RSI']=100-(100/(1+rs))
        return df
    
    def add_trend(self,df,window=20):
        """
        calculate average price of past 20 min
        """
        df['Log_Returns']=np.log(df['Close']/df['Close'].shift(1))
        df['Volatility']=df['Log_Returns'].rolling(window=10).std()
        
        
        df['SMA_20']=df['Close'].rolling(window=window).mean()
        df['std_dev']=df['Close'].rolling(window=window).std()
        df['up_bound']=df['SMA_20']+(df['std_dev']*2)
        df['low_bound']=df['SMA_20']-(df['std_dev']*2)
        # Distance from SMA: Tells us if the price is 'stretched' far from the average
        df['Price_Dist_SMA'] = df['Close'] / df['SMA_20']
        df['b_band_pos']=(df['Close']-df['low_bound'])/(df['up_bound']-df['low_bound'])
        
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']        
        return df.dropna()