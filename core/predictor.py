from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

class MarketPredictor:
    """
    Module 4: Predictor Main Brain
    Work: Train a model to predict if the price will be higher in 5 minutes.
    """
    def __init__(self):
        # CRITICAL: These must match the columns created in FeatureEngineer
        self.feature_cols = [
            'Log_Returns', 
            'Volatility', 
            'RSI', 
            'Price_Dist_SMA', 
            'b_band_pos',   # Added from Module 3
            'MACD',         # Added from Module 3
            'MACD_Hist'     # Added from Module 3
        ]
        
        # Hyperparameters tuned for high-noise 1m data
        self.model = RandomForestClassifier(
            n_estimators=150,     # More trees for a more stable 'consensus'
            max_depth=6,          # Kept shallow to prevent memorizing noise
            min_samples_leaf=12,  # Requires more evidence to create a rule
            max_features='sqrt', 
            random_state=42,
            class_weight='balanced' # Fixes issues if the market is mostly flat
        )
        
    def prepare_target(self, df):
        """
        Creating label: 1 if price is higher in 5 minutes (5 rows ahead)
        """
        df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
        return df.dropna()
    
    def train(self, df):
        X = df[self.feature_cols]
        y = df['Target']
        
        # Time-Series Split: Shuffle=False ensures we don't 'peek' into the future
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        if len(np.unique(y_train)) < 2:
            print("⚠️ MarketPulse AI: Not enough variety in data. Skipping...")
            return 0.0
        
        self.model.fit(X_train, y_train)
        
        # This accuracy is now 'Honest Backtesting' accuracy
        accuracy = self.model.score(X_test, y_test)
        print(f"✅ Training Complete. Backtest Accuracy: {accuracy:.2%}")
        return accuracy