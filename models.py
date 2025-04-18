import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_model(X_train, y_train, model_type='random_forest'):
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(models_dir, f'{model_type}.pkl'))
    return model

def predict(X, model_type='random_forest'):
    model_path = os.path.join('models', f'{model_type}.pkl')
    try:
        model = joblib.load(model_path)
        return model.predict(X)
    except:
        raise Exception("Model not found. Please train the model first.") 