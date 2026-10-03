import argparse
import pickle
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Inference CLI for Bike Sharing Predictive Tool")
    # Define required structural parameters matching feature expectations
    parser.add_argument('--temp', type=float, required=True, help="Normalized temperature metric (0.0 to 1.0)")
    parser.add_argument('--humidity', type=float, required=True, help="Normalized humidity metric (0.0 to 1.0)")
    parser.add_argument('--windspeed', type=float, required=True, help="Normalized windspeed metric (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    # Load serialized model pipeline
    with open('config/model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    # Construct instant payload
    payload = pd.DataFrame([[args.temp, args.humidity, args.windspeed]], 
                           columns=['temp', 'hum', 'windspeed'])
    
    prediction = model.predict(payload)[0]
    print(f"\n🚀 [Prediction Result] Estimated Demand Count: {int(prediction)} units\n")

if __name__ == '__main__':
    main()