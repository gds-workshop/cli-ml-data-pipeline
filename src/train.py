import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model():
    print("🏋️ Training the predictive model...")
    
    # 1. Load the processed data splits
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed split files not found. Run preprocess.py first!")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 2. Separate features and target
    X_train = train_df.drop(columns=['cnt'])
    y_train = train_df['cnt']
    X_test = test_df.drop(columns=['cnt'])
    y_test = test_df['cnt']
    
    # 3. Fit a Machine Learning Model (RandomForest handles complex non-linear trends well)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 4. Evaluate Model Performance
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n📊 --- Baseline Performance Metrics ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared Score (R²):     {r2:.4f}")
    print("---------------------------------------\n")
    
    # 5. Serialize and Save the Model object to disk
    os.makedirs("config", exist_ok=True)
    model_pickle_path = "config/model.pkl"
    
    with open(model_pickle_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"💾 Model successfully serialized and saved to: {model_pickle_path}")

if __name__ == "__main__":
    train_model()