import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)

    def detect_feature_types(self, df):
        """Detect numerical and categorical features"""
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return numerical_features, categorical_features

    def create_preprocessor(self, numerical_features, categorical_features):
        """Create preprocessing pipeline"""
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        return self.preprocessor

    def fit_transform(self, df):
        """Fit and transform the data"""
        numerical_features, categorical_features = self.detect_feature_types(df)
        self.create_preprocessor(numerical_features, categorical_features)
        transformed_data = self.preprocessor.fit_transform(df)
        
        # Save preprocessor
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
        joblib.dump(self.preprocessor, preprocessor_path)
        
        return transformed_data

    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        if self.preprocessor is None:
            preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
            self.preprocessor = joblib.load(preprocessor_path)
        return self.preprocessor.transform(df)

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []

    def add_temporal_features(self, df, date_column):
        """Add temporal features from date column"""
        df[date_column] = pd.to_datetime(df[date_column])
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['dayofweek'] = df[date_column].dt.dayofweek
        df['hour'] = df[date_column].dt.hour
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        return df

    def add_weather_features(self, df):
        """Add weather-related features"""
        # Example: Add temperature categories
        df['temp_category'] = pd.cut(df['temperature'], 
                                   bins=[-np.inf, 0, 10, 20, 30, np.inf],
                                   labels=['very_cold', 'cold', 'mild', 'warm', 'hot'])
        return df

    def add_traffic_features(self, df):
        """Add traffic-related features"""
        # Example: Add traffic density categories
        df['traffic_density'] = pd.cut(df['traffic_volume'],
                                     bins=[-np.inf, 100, 500, 1000, np.inf],
                                     labels=['very_light', 'light', 'moderate', 'heavy'])
        return df

    def add_location_features(self, df):
        """Add location-related features"""
        # Example: Add distance to nearest city center
        df['distance_to_center'] = np.sqrt(
            (df['latitude'] - df['city_center_lat'])**2 +
            (df['longitude'] - df['city_center_lon'])**2
        )
        return df

    def add_interaction_features(self, df):
        """Add interaction features between variables"""
        # Example: Interaction between weather and traffic
        df['weather_traffic_interaction'] = df['temperature'] * df['traffic_volume']
        return df

    def engineer_features(self, df, date_column=None):
        """Apply all feature engineering steps"""
        if date_column:
            df = self.add_temporal_features(df, date_column)
        df = self.add_weather_features(df)
        df = self.add_traffic_features(df)
        df = self.add_location_features(df)
        df = self.add_interaction_features(df)
        self.feature_names = df.columns.tolist()
        return df

class DataSplitter:
    def __init__(self, test_size=0.2, val_size=0.2, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        from sklearn.model_selection import train_test_split
        
        # First split: separate test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Second split: separate validation set from remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.val_size/(1-self.test_size),
            random_state=self.random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test 