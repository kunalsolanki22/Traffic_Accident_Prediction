# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import requests

# Initialize global variables
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
model = None
scaler = StandardScaler()
label_encoders = {}

# Initialize geocoder
geolocator = Nominatim(user_agent="accident_prediction")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# List of major Indian cities
INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", 
    "Pune", "Ahmedabad", "Surat", "Jaipur", "Lucknow", "Kanpur", 
    "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam", "Patna", 
    "Vadodara", "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Faridabad", 
    "Meerut", "Rajkot", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad"
]

def get_city_coordinates(city_name):
    """Get coordinates for a given city name"""
    try:
        # Append ", India" to ensure we get Indian cities
        location = geocode(f"{city_name}, India")
        if location:
            return location.latitude, location.longitude
        return None
    except Exception as e:
        print(f"Error getting coordinates for {city_name}: {str(e)}")
        return None

def get_traffic_data(lat, lon):
    """Get traffic data from TomTom API"""
    api_key = os.getenv('TOMTOM_API_KEY')
    if not api_key:
        print("TomTom API key not found. Please set the TOMTOM_API_KEY environment variable.")
        return None
    
    base_url = "https://api.tomtom.com/traffic/services/4"
    endpoint = f"{base_url}/flowSegmentData/absolute/10/json"
    
    params = {
        'key': api_key,
        'point': f"{lat},{lon}",
        'unit': 'KMPH'
    }
    
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'flowSegmentData' in data:
            flow_data = data['flowSegmentData']
            return {
                'current_speed': flow_data.get('currentSpeed', 0),
                'free_flow_speed': flow_data.get('freeFlowSpeed', 0),
                'confidence': flow_data.get('confidence', 0)
            }
        return None
    except Exception as e:
        print(f"Error fetching traffic data: {str(e)}")
        return None

def predict_with_location(features, city_name):
    """Make predictions with real-time traffic data from location"""
    # Get coordinates for the city
    coords = get_city_coordinates(city_name)
    if not coords:
        print(f"Could not find coordinates for {city_name}")
        return predict(features)
    
    lat, lon = coords
    print(f"\nCoordinates for {city_name}: {lat}, {lon}")
    
    # Get traffic data
    traffic_data = get_traffic_data(lat, lon)
    if traffic_data:
        print("\nReal-time traffic data:")
        print(f"Current Speed: {traffic_data['current_speed']} KMPH")
        print(f"Free Flow Speed: {traffic_data['free_flow_speed']} KMPH")
        print(f"Confidence: {traffic_data['confidence']}%")
        
        # Update features with traffic data
        features = list(features)  # Convert to list if it's not already
        features.extend([
            traffic_data['current_speed'],
            traffic_data['free_flow_speed'],
            traffic_data['confidence']
        ])
    else:
        print("\nNo traffic data available. Using default values.")
        features = list(features)
        features.extend([0, 0, 0])  # Default values
    
    return predict(features)

def load_data(file_path):
    """Load and preprocess data"""
    df = pd.read_csv(file_path)
    
    # Convert categorical columns to numeric using label encoding
    categorical_columns = [
        'Day_of_week', 'Age_band_of_driver', 'Sex_of_driver', 
        'Educational_level', 'Vehicle_driver_relation', 'Type_of_vehicle',
        'Area_accident_occured', 'Road_allignment', 'Types_of_Junction',
        'Road_surface_type', 'Road_surface_conditions', 'Light_conditions',
        'Weather_conditions', 'Type_of_collision', 'Vehicle_movement',
        'Cause_of_accident', 'Accident_severity'
    ]
    
    # Convert experience ranges to numeric values
    experience_mapping = {
        'Above 10yr': 10,
        '5-10yr': 7.5,
        '2-5yr': 3.5,
        '1-2yr': 1.5,
        'Below 1yr': 0.5,
        'No Licence': 0,
        'unknown': 0
    }
    
    if 'Driving_experience' in df.columns:
        df['Driving_experience'] = df['Driving_experience'].map(experience_mapping)
    
    # Convert service year ranges to numeric values
    service_mapping = {
        'Above 10yr': 10,
        '5-10yr': 7.5,
        '2-5yr': 3.5,
        '1-2yr': 1.5,
        'Below 1yr': 0.5,
        'unknown': 0
    }
    
    if 'Service_year_of_vehicle' in df.columns:
        df['Service_year_of_vehicle'] = df['Service_year_of_vehicle'].map(service_mapping)
    
    for col in categorical_columns:
        if col in df.columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col].astype(str))
    
    return df

def prepare_features(df):
    """Prepare features for modeling"""
    # Select relevant features
    features = [
        'Day_of_week', 'Age_band_of_driver', 'Driving_experience',
        'Service_year_of_vehicle', 'Number_of_vehicles_involved',
        'Number_of_casualties', 'Light_conditions', 'Weather_conditions',
        'Road_surface_conditions', 'Type_of_collision'
    ]
    
    target = 'Accident_severity'
    
    # Fill any missing values with 0
    for feature in features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(0)
    
    X = df[features]
    y = df[target]
    
    # Scale features
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, features

def train_model(X, y):
    """Train the model"""
    global model
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and scaler
    joblib.dump(model, os.path.join(models_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    
    return X_test, y_test

def evaluate_model(X_test, y_test, feature_names):
    """Evaluate model performance"""
    global model
    
    if model is None:
        model = joblib.load(os.path.join(models_dir, 'model.pkl'))
    
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.show()
    
    return feature_importance

def predict(features):
    """Make predictions"""
    global model, scaler
    
    if model is None:
        model = joblib.load(os.path.join(models_dir, 'model.pkl'))
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    # Convert numeric prediction back to label
    severity_label = label_encoders.get('Accident_severity').inverse_transform([prediction])[0]
    
    # Convert numeric class labels back to original labels
    class_labels = label_encoders.get('Accident_severity').inverse_transform(model.classes_)
    probabilities = pd.Series(probability, index=class_labels)
    
    return severity_label, probabilities

# Main execution
if __name__ == "__main__":
    # Load the data
    data_file = os.path.join('data', 'Road.csv')
    df = load_data(data_file)
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Train and evaluate the model
    X_test, y_test = train_model(X, y)
    feature_importance = evaluate_model(X_test, y_test, feature_names)
    
    # Example prediction with city
    example_features = [
        label_encoders['Day_of_week'].transform(['Friday'])[0],  # Day of week
        label_encoders['Age_band_of_driver'].transform(['18-30'])[0],  # Age band
        10,  # Driving experience (Above 10yr)
        3.5,  # Service years (2-5yr)
        2,  # Number of vehicles
        1,  # Number of casualties
        label_encoders['Light_conditions'].transform(['Darkness - lights lit'])[0],  # Light conditions
        label_encoders['Weather_conditions'].transform(['Normal'])[0],  # Weather
        label_encoders['Road_surface_conditions'].transform(['Dry'])[0],  # Road conditions
        label_encoders['Type_of_collision'].transform(['Vehicle with vehicle collision'])[0]  # Collision type
    ]
    
    # Select a city
    print("\nAvailable Indian cities:")
    for i, city in enumerate(INDIAN_CITIES, 1):
        print(f"{i}. {city}")
    
    city_name = input("\nEnter city name from the list above: ")
    if city_name in INDIAN_CITIES:
        severity, probabilities = predict_with_location(example_features, city_name)
        print(f"\nPredicted Accident Severity for {city_name}: {severity}")
        print("\nProbability Distribution:")
        print(probabilities)
    else:
        print("Invalid city name. Please choose from the list of available cities.") 