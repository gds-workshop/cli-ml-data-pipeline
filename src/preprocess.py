import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    print("🔄 Starting data preprocessing pipeline...")
    
    # 1. Load the raw dataset
    raw_path = "data/raw/hour.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing raw data at {raw_path}. Run fetch_data.sh first!")
    
    df = pd.read_csv(raw_path)
    
    # 2. Feature Selection: Drop columns that don't help predict future demand
    # 'instant' is just an index. 'dteday' is a string date (we use season/hr/mnth instead).
    # 'casual' and 'registered' add up exactly to our target 'cnt', causing data leakage.
    columns_to_drop = ['instant', 'dteday', 'casual', 'registered']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # 3. Handle Missing Values explicitly (Even if there are none, this ensures pipeline safety)
    df = df.dropna()
    
    # 4. Separate features (X) and target variable (y)
    X = df.drop(columns=['cnt'])
    y = df['cnt']
    
    # 5. Split into Train/Test sets (80% training, 20% testing for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Save the split datasets as processed CSV files
    os.makedirs("data/processed", exist_ok=True)
    
    # Recombine temporarily just to save clean, split CSV files
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv("data/processed/train.csv", index=False)
    test_data.to_csv("data/processed/test.csv", index=False)
    
    print("✅ Preprocessing complete! Training and testing splits saved to data/processed/")

if __name__ == "__main__":
    preprocess_data()