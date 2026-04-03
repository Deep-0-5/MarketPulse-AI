from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

class MarketPredictor:
    """
    module 4: predictor main brain
    work:train a model to predict if next minute will be up or down
    """
    def __init__(self):
        self.model=RandomForestClassifier(n_estimators=100,random_state=42)
        self.feature_cols=['Log_Returns', 'Volatility', 'RSI', 'Price_Dist_SMA']
        
    def prepare_target(self,df):
        """
        creating label if price goes high then label=1 else lable=0
        """
        df['Target']=(df['Close'].shift(-5) > df['Close']).astype(int)
        
        return df.dropna()
    
    def train(self,df):
        #we are giving clues to the AI
        # feature=['Log_Returns', 'Volatility', 'RSI', 'Price_Dist_SMA']
        X=df[self.feature_cols]
        y=df['Target']
        
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
        
        if len(np.unique(y_train)) < 2:
            print("⚠️ MarketPulse AI: Not enough variety in data to train yet. Skipping...")
            return 0.0
        
        print("🧠 MarketPulse AI: Training the Random Forest model...")
        self.model.fit(X_train,y_train)
        
        accuracy=self.model.score(X_test,y_test)
        print(f"✅ Training Complete. Model Accuracy: {accuracy:.2%}")
        return accuracy