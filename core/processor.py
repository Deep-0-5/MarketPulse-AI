import numpy as np
import pandas as pd

class DataProcessor:
    """
    Module 2: data processor to peoccess fetrhced data
    work: clean data and handle maths 
    """
    def clean_data(self,df):
        initial_len=len(df)
        df=df.dropna()
        df=df[df['Volume']>0]
        
        print(f"🧹 MarketPulse AI: Cleaned {initial_len - len(df)} bad/static rows.")
        return df
    
    def add_feature(self,df):
        # calculate market movement
        #log returns - standrize data for model
        df['Log_Returns']=np.log(df['Close']/df['Close'].shift(1))
        
        #volatility-calculate sd(standard deveition)
        df['Volatility']=df['Log_Returns'].rolling(window=10).std()
        
        return df.dropna()
        
