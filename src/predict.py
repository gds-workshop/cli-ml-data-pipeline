import argparse
import os
import pickle
import pandas as pd

def main():
    # 1. Establish the interactive CLI interface
    parser = argparse.ArgumentParser(description="Production CLI Tool for Bike Sharing Demand Predictions")
    
    # 2. Map every feature your model expects to a command-line argument flag
    parser.add_argument('--season', type=int, required=True, help="Season (1:spring, 2:summer, 3:fall, 4:winter)")
    parser.add_argument('--yr', type=int, required=True, help="Year offset (0: 2011, 1: 2012)")
    parser.add_argument('--mnth', type=int, required=True, help="Month (1 to 12)")
    parser.add_argument('--hr', type=int, required=True, help="Hour of the day (0 to 23)")
    parser.add_argument('--holiday', type=int, required=True, help="Is holiday? (0 or 1)")
    parser.add_argument('--weekday', type=int, required=True, help="Day of the week (0 to 6)")
    parser.add_argument('--workingday', type=int, required=True, help="If day is neither weekend nor holiday (0 or 1)")
    parser.add_argument('--weathersit', type=int, required=True, help="Weather status rating (1 to 4)")
    parser.add_argument('--temp', type=float, required=True, help="Normalized temperature (0.0 to 1.0)")
    parser.add_argument('--atemp', type=float, required=True, help="Normalized 'feels like' temperature (0.0 to 1.0)")
    parser.add_argument('--hum', type=float, required=True, help="Normalized humidity (0.0 to 1.0)")
    parser.add_argument('--windspeed', type=float, required=True, help="Normalized windspeed (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    # 3. Safely load the static model payload
    model_path = 'config/model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first!")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # 4. Format the terminal flags exactly into a structured DataFrame row
    input_data = pd.DataFrame([{
        'season': args.season, 'yr': args.yr, 'mnth': args.mnth, 'hr': args.hr,
        'holiday': args.holiday, 'weekday': args.weekday, 'workingday': args.workingday,
        'weathersit': args.weathersit, 'temp': args.temp, 'atemp': args.atemp,
        'hum': args.hum, 'windspeed': args.windspeed
    }])
    
    # 5. Generate prediction output
    prediction = model.predict(input_data)[0]
    print(f"\n🚀 [Prediction Result] Estimated Demand Count: {int(max(0, prediction))} bikes/hour\n")

if __name__ == '__main__':
    main()