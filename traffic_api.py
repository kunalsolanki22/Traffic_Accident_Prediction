import requests
import pandas as pd
from typing import Dict, List, Optional
import os
from datetime import datetime

class TomTomTrafficAPI:
    def __init__(self, api_key: str):
        """
        Initialize the TomTom Traffic API client.
        
        Args:
            api_key (str): Your TomTom API key
        """
        self.api_key = api_key
        self.base_url = "https://api.tomtom.com/traffic/services/4"
        
    def get_traffic_flow(self, 
                        lat: float, 
                        lon: float,
                        radius: int = 100) -> Dict:
        """
        Get traffic flow data for a specific location.
        
        Args:
            lat (float): Latitude of the location
            lon (float): Longitude of the location
            radius (int): Search radius in meters (default: 100)
            
        Returns:
            Dict: Traffic flow data
        """
        # Validate coordinates
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError("Invalid latitude or longitude values")

        endpoint = f"{self.base_url}/flowSegmentData/absolute/10/json"
        
        params = {
            'key': self.api_key,
            'point': f"{lat},{lon}",
            'unit': 'KMPH',
            'thickness': 1,
            'radius': radius
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()  # Raise an error for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch traffic data: {str(e)}")
    
    def get_incidents(self,
                     bbox: str,
                     incident_type: Optional[str] = None) -> List[Dict]:
        """
        Get traffic incidents in a specific bounding box.
        
        Args:
            bbox (str): Bounding box in format "minLon,minLat,maxLon,maxLat"
            incident_type (str, optional): Filter by incident type
            
        Returns:
            List[Dict]: List of traffic incidents
        """
        endpoint = f"{self.base_url}/incidentDetails"
        params = {
            'bbox': bbox,
            'key': self.api_key,
            'language': 'en-GB',
            'categoryFilter': incident_type if incident_type else '0,1,2,3,4,5,6,7,8,9,10,11,14'
        }
        
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json().get('incidents', [])
    
    def get_traffic_flow_dataframe(self, 
                                 lat: float, 
                                 lon: float,
                                 radius: int = 100) -> pd.DataFrame:
        """
        Get traffic flow data as a pandas DataFrame.
        
        Args:
            lat (float): Latitude of the location
            lon (float): Longitude of the location
            radius (int): Search radius in meters
            
        Returns:
            pd.DataFrame: Traffic flow data in DataFrame format
        """
        try:
            data = self.get_traffic_flow(lat, lon, radius)
            
            # Extract flow data
            if 'flowSegmentData' not in data:
                return pd.DataFrame({
                    'timestamp': [datetime.now()],
                    'current_speed': [0],
                    'free_flow_speed': [0],
                    'confidence': [0],
                    'coordinates': [f"{lat},{lon}"]
                })
                
            flow_data = data['flowSegmentData']
            
            return pd.DataFrame([{
                'timestamp': datetime.now(),
                'current_speed': flow_data.get('currentSpeed', 0),
                'free_flow_speed': flow_data.get('freeFlowSpeed', 0),
                'confidence': flow_data.get('confidence', 0),
                'coordinates': f"{lat},{lon}"
            }])
            
        except Exception as e:
            raise Exception(f"Failed to process traffic data: {str(e)}") 