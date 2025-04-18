import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from dotenv import load_dotenv
from traffic_api import TomTomTrafficAPI

load_dotenv()

# Basic setup
model = None
scaler = StandardScaler()
label_encoders = {}
geolocator = Nominatim(user_agent="accident_prediction_app")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# Major Indian cities
INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", 
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Kolkata"
]

# Initialize TomTom API
api_key = os.getenv('TOMTOM_API_KEY')
if not api_key:
    st.error("TomTom API key not found in .env file")
    st.stop()
tomtom_api = TomTomTrafficAPI(api_key=api_key)

def get_traffic_data(city):
    try:
        location = geocode(f"{city}, India")
        if location:
            traffic = tomtom_api.get_traffic_flow_dataframe(location.latitude, location.longitude)
            return {
                'current_speed': float(traffic['current_speed'].iloc[0]),
                'free_flow_speed': float(traffic['free_flow_speed'].iloc[0]),
                'confidence': float(traffic['confidence'].iloc[0])
            }
    except Exception as e:
        st.error(f"Error getting traffic data: {e}")
    return {'current_speed': 0, 'free_flow_speed': 0, 'confidence': 0}

def process_data(df):
    # Convert categorical columns
    cat_columns = ['Day_of_week', 'Age_band_of_driver', 'Light_conditions', 
                  'Weather_conditions', 'Road_surface_conditions', 
                  'Type_of_collision', 'Accident_severity']
    
    # Convert experience and service years
    value_maps = {
        'Above 10yr': 10, '5-10yr': 7.5, '2-5yr': 3.5,
        '1-2yr': 1.5, 'Below 1yr': 0.5, 'No Licence': 0, 'unknown': 0
    }
    
    df['Driving_experience'] = df['Driving_experience'].map(value_maps)
    df['Service_year_of_vehicle'] = df['Service_year_of_vehicle'].map(value_maps)
    
    for col in cat_columns:
        if col in df.columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col].astype(str))
    
    return df

def train_predict_model(df):
    features = [
        'Day_of_week', 'Age_band_of_driver', 'Driving_experience',
        'Service_year_of_vehicle', 'Number_of_vehicles_involved',
        'Light_conditions', 'Weather_conditions', 'Road_surface_conditions', 
        'Type_of_collision', 'current_speed', 'free_flow_speed', 'confidence'
    ]
    
    # Add traffic features
    for col in ['current_speed', 'free_flow_speed', 'confidence']:
        df[col] = 0
    
    X = df[features]
    y = df['Accident_severity']
    
    # Scale features
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    global model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    st.bar_chart(importance.set_index('Feature'))

def main():
    st.title("Road Accident Severity Prediction")
    
    # Load and process data
    df = pd.read_csv('data/Road.csv')
    df = process_data(df)
    
    # Train model button
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            train_predict_model(df)
    
    # Prediction interface
    st.header("Make a Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("Select City", INDIAN_CITIES)
        day = st.selectbox("Day of Week", label_encoders['Day_of_week'].classes_)
        age = st.selectbox("Driver Age Band", label_encoders['Age_band_of_driver'].classes_)
        exp = st.selectbox("Driving Experience", ['Above 10yr', '5-10yr', '2-5yr', '1-2yr', 'Below 1yr'])
        service = st.selectbox("Vehicle Age", ['Above 10yr', '5-10yr', '2-5yr', '1-2yr', 'Below 1yr'])
    
    with col2:
        vehicles = st.number_input("Vehicles Involved", 1, 10, 2)
        light = st.selectbox("Light Conditions", label_encoders['Light_conditions'].classes_)
        weather = st.selectbox("Weather", label_encoders['Weather_conditions'].classes_)
        road = st.selectbox("Road Conditions", label_encoders['Road_surface_conditions'].classes_)
        collision = st.selectbox("Collision Type", label_encoders['Type_of_collision'].classes_)
    
    if st.button("Predict"):
        # Get traffic data
        traffic = get_traffic_data(city)
        st.write(f"Traffic Data for {city}:")
        st.write(f"Current Speed: {traffic['current_speed']} KMPH")
        st.write(f"Free Flow Speed: {traffic['free_flow_speed']} KMPH")
        
        # Prepare features
        value_maps = {
            'Above 10yr': 10, '5-10yr': 7.5, '2-5yr': 3.5,
            '1-2yr': 1.5, 'Below 1yr': 0.5
        }
        
        features = [
            label_encoders['Day_of_week'].transform([day])[0],
            label_encoders['Age_band_of_driver'].transform([age])[0],
            value_maps[exp],
            value_maps[service],
            vehicles,
            label_encoders['Light_conditions'].transform([light])[0],
            label_encoders['Weather_conditions'].transform([weather])[0],
            label_encoders['Road_surface_conditions'].transform([road])[0],
            label_encoders['Type_of_collision'].transform([collision])[0],
            traffic['current_speed'],
            traffic['free_flow_speed'],
            traffic['confidence']
        ]
        
        # Load model if needed
        global model, scaler
        if model is None:
            try:
                model = joblib.load('models/model.pkl')
                scaler = joblib.load('models/scaler.pkl')
            except:
                st.error("Please train the model first")
                return
        
        # Make prediction
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        
        # Show results
        severity = label_encoders['Accident_severity'].inverse_transform([prediction])[0]
        st.write(f"Predicted Severity: {severity}")
        
        classes = label_encoders['Accident_severity'].inverse_transform(model.classes_)
        st.write("Probability Distribution:")
        st.bar_chart(pd.Series(proba, index=classes))

if __name__ == "__main__":
    main() 