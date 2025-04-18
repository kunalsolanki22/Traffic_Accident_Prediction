# Road Accident Severity Prediction

A machine learning application that predicts the severity of road accidents in Indian cities using historical accident data and real-time traffic information from the TomTom API.

## Features

- Real-time traffic data integration using TomTom API
- Machine learning model for accident severity prediction
- Interactive web interface using Streamlit
- Support for 10 major Indian cities
- Feature importance visualization
- Probability distribution for predictions

## Prerequisites

- Python 3.7+
- TomTom API Key (get it from [TomTom Developer Portal](https://developer.tomtom.com/))

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd accident-prediction
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your TomTom API key:
```
TOMTOM_API_KEY=your_api_key_here
```

5. Create a `data` directory and place your accident dataset (`Road.csv`) in it:
```bash
mkdir data
# Copy your Road.csv file to the data directory
```

## Project Structure

```
accident-prediction/
├── app.py              # Main Streamlit application
├── models.py           # Machine learning model functions
├── traffic_api.py      # TomTom API integration
├── requirements.txt    # Project dependencies
├── .env               # Environment variables
├── data/              # Data directory
│   └── Road.csv       # Accident dataset
└── models/            # Saved model files
    ├── model.pkl      # Trained model
    └── scaler.pkl     # Feature scaler
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and go to `http://localhost:8501`

3. Using the application:
   - Click "Train Model" to train the prediction model
   - Select a city and input accident parameters
   - Click "Predict" to get the severity prediction

## Features Used for Prediction

- Day of Week
- Driver Age Band
- Driving Experience
- Vehicle Age
- Number of Vehicles Involved
- Light Conditions
- Weather Conditions
- Road Surface Conditions
- Type of Collision
- Real-time Traffic Data:
  - Current Speed
  - Free Flow Speed
  - Traffic Confidence

## Model Details

- Algorithm: Random Forest Classifier
- Features: 12 input features (9 accident-related + 3 traffic-related)
- Training: 80% training data, 20% testing data
- Evaluation: Accuracy score and feature importance visualization

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TomTom API for providing real-time traffic data
- Streamlit for the web interface framework
- scikit-learn for machine learning tools 