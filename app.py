from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Global variables to store loaded models
rf_model = None
lr_model = None
feature_columns = None
model_metadata = None

def load_models():
    """Load the saved models and feature columns."""
    global rf_model, lr_model, feature_columns, model_metadata
    
    try:
        rf_model = joblib.load('rf_model.pkl')
        lr_model = joblib.load('lr_model.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        model_metadata = joblib.load('model_metadata.pkl')
        print("Models loaded successfully!")
        return True
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Please run main.py first to train and save the models.")
        return False

def prepare_features(hour, day_of_week, month, year=None, day_of_year=None, week_of_year=None,
                    is_weekend=None, is_rush_hour=None, is_night=None, is_morning=None, 
                    is_afternoon=None, is_evening=None, season=None, day_of_month=1,
                    borough=None, zip_code=None, vehicle_types=None, contributing_factors=None,
                    num_vehicles=1, has_fatalities=False, has_pedestrian_injuries=False,
                    has_cyclist_injuries=False, has_motorist_injuries=False):
    """
    Prepare feature vector for prediction.
    
    Parameters:
    - hour: int (0-23)
    - day_of_week: int (0=Monday, 6=Sunday)
    - month: int (1-12)
    - year: int (optional, defaults to 2024)
    - day_of_year: int (1-365, optional, calculated if not provided)
    - week_of_year: int (1-52, optional, calculated if not provided)
    - is_weekend: bool (optional, calculated from day_of_week if not provided)
    - is_rush_hour: bool (optional, calculated from hour if not provided)
    - is_night, is_morning, is_afternoon, is_evening: bool (optional, calculated from hour)
    - season: int (0=Winter, 1=Spring, 2=Summer, 3=Fall, optional, calculated from month)
    - day_of_month: int (1-31, default=1)
    - borough: str (optional)
    - zip_code: str (optional)
    - vehicle_types: list of str (optional)
    - contributing_factors: list of str (optional)
    - num_vehicles: int (default=1)
    - has_fatalities, has_pedestrian_injuries, etc.: bool (default=False)
    """
    # Set defaults
    if year is None:
        year = 2024
    if day_of_year is None:
        # Approximate calculation
        day_of_year = (month - 1) * 30 + day_of_month
    if week_of_year is None:
        week_of_year = day_of_year // 7 + 1
    if is_weekend is None:
        is_weekend = (day_of_week >= 5)
    if is_rush_hour is None:
        is_rush_hour = ((hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19))
    if season is None:
        season = ((month % 12) // 3)
    
    # Calculate time periods
    if is_night is None:
        is_night = (hour >= 22 or hour <= 5)
    if is_morning is None:
        is_morning = (hour >= 6 and hour <= 11)
    if is_afternoon is None:
        is_afternoon = (hour >= 12 and hour <= 17)
    if is_evening is None:
        is_evening = (hour >= 18 and hour <= 21)
    
    # Create base features dictionary
    features = {
        'HOUR': [hour],
        'DAY_OF_WEEK': [day_of_week],
        'MONTH': [month],
        'YEAR': [year],
        'DAY_OF_YEAR': [day_of_year],
        'WEEK_OF_YEAR': [week_of_year],
        'IS_WEEKEND': [1 if is_weekend else 0],
        'IS_RUSH_HOUR': [1 if is_rush_hour else 0],
        'IS_NIGHT': [1 if is_night else 0],
        'IS_MORNING': [1 if is_morning else 0],
        'IS_AFTERNOON': [1 if is_afternoon else 0],
        'IS_EVENING': [1 if is_evening else 0],
        'SEASON': [season],
        'HOUR_SIN': [np.sin(2 * np.pi * hour / 24)],
        'HOUR_COS': [np.cos(2 * np.pi * hour / 24)],
        'DAY_OF_WEEK_SIN': [np.sin(2 * np.pi * day_of_week / 7)],
        'DAY_OF_WEEK_COS': [np.cos(2 * np.pi * day_of_week / 7)],
        'MONTH_SIN': [np.sin(2 * np.pi * month / 12)],
        'MONTH_COS': [np.cos(2 * np.pi * month / 12)],
        'DAY_OF_MONTH_SIN': [np.sin(2 * np.pi * day_of_month / 31)],
        'DAY_OF_MONTH_COS': [np.cos(2 * np.pi * day_of_month / 31)],
        'DAY_OF_YEAR_SIN': [np.sin(2 * np.pi * day_of_year / 365)],
        'DAY_OF_YEAR_COS': [np.cos(2 * np.pi * day_of_year / 365)],
        'WEEK_OF_YEAR_SIN': [np.sin(2 * np.pi * week_of_year / 52)],
        'WEEK_OF_YEAR_COS': [np.cos(2 * np.pi * week_of_year / 52)],
        'RUSH_HOUR_WEEKEND': [1 if (is_rush_hour and is_weekend) else 0],
        'NIGHT_WEEKEND': [1 if (is_night and is_weekend) else 0],
        'MORNING_WEEKEND': [1 if (is_morning and is_weekend) else 0],
        'NUM_VEHICLES': [num_vehicles],
        'HAS_FATALITIES': [1 if has_fatalities else 0],
        'HAS_PEDESTRIAN_INJURIES': [1 if has_pedestrian_injuries else 0],
        'HAS_CYCLIST_INJURIES': [1 if has_cyclist_injuries else 0],
        'HAS_MOTORIST_INJURIES': [1 if has_motorist_injuries else 0],
    }
    
    # Add aggregate features (use default/mean values)
    features['AVG_INJURIES_BY_HOUR'] = [0.5]
    features['AVG_INJURIES_BY_DAY_OF_WEEK'] = [0.5]
    features['AVG_INJURIES_BY_MONTH'] = [0.5]
    if 'AVG_INJURIES_BY_BOROUGH' in feature_columns:
        features['AVG_INJURIES_BY_BOROUGH'] = [0.5]
    
    # Create DataFrame with all feature columns, defaulting to 0
    df = pd.DataFrame(index=[0])
    for col in feature_columns:
        if col in features:
            df[col] = features[col]
        else:
            df[col] = [0]  # Default to 0 for features not provided
    
    # Ensure columns are in the correct order
    df = df[feature_columns]
    
    return df

@app.route('/')
def home():
    """Home endpoint with API information."""
    return jsonify({
        'message': 'Motor Vehicle Collision Injury Prediction API',
        'endpoints': {
            '/predict': 'POST - Make injury predictions',
            '/health': 'GET - Check API health and model status'
        },
        'usage': {
            'endpoint': '/predict',
            'method': 'POST',
            'required_parameters': {
                'hour': 'int (0-23)',
                'day_of_week': 'int (0=Monday, 6=Sunday)',
                'month': 'int (1-12)'
            },
            'optional_parameters': {
                'year': 'int (default: 2024)',
                'day_of_month': 'int (default: 1)',
                'num_vehicles': 'int (default: 1)'
            }
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    if rf_model is None or lr_model is None:
        return jsonify({
            'status': 'error',
            'message': 'Models not loaded. Please run main.py first to train and save models.'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'model_performance': {
            'random_forest': {
                'r2_score': model_metadata['rf_test_r2'],
                'mae': model_metadata['rf_test_mae']
            },
            'linear_regression': {
                'r2_score': model_metadata['lr_test_r2'],
                'mae': model_metadata['lr_test_mae']
            }
        },
        'num_features': model_metadata['num_features']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the number of persons injured based on time features.
    
    Required parameters:
    - hour: int (0-23)
    - day_of_week: int (0=Monday, 6=Sunday)
    - month: int (1-12)
    
    Optional parameters:
    - year: int (default: 2024)
    - day_of_month: int (default: 1)
    - num_vehicles: int (default: 1)
    """
    # Check if models are loaded
    if rf_model is None or lr_model is None:
        return jsonify({
            'error': 'Models not loaded. Please run main.py first to train and save models.'
        }), 503
    
    # Get JSON data from request
    data = request.get_json()
    
    if not data:
        return jsonify({
            'error': 'No JSON data provided. Please send a JSON object with required parameters.'
        }), 400
    
    # Validate required parameters
    required_params = ['hour', 'day_of_week', 'month']
    missing_params = [param for param in required_params if param not in data]
    
    if missing_params:
        return jsonify({
            'error': f'Missing required parameters: {", ".join(missing_params)}'
        }), 400
    
    # Extract and validate parameters
    try:
        hour = int(data['hour'])
        day_of_week = int(data['day_of_week'])
        month = int(data['month'])
        
        # Validate ranges
        if not (0 <= hour <= 23):
            return jsonify({'error': 'hour must be between 0 and 23'}), 400
        if not (0 <= day_of_week <= 6):
            return jsonify({'error': 'day_of_week must be between 0 (Monday) and 6 (Sunday)'}), 400
        if not (1 <= month <= 12):
            return jsonify({'error': 'month must be between 1 and 12'}), 400
        
        # Extract optional parameters
        year = int(data.get('year', 2024))
        day_of_month = int(data.get('day_of_month', 1))
        num_vehicles = int(data.get('num_vehicles', 1))
        
        # Validate optional parameters
        if not (1 <= day_of_month <= 31):
            return jsonify({'error': 'day_of_month must be between 1 and 31'}), 400
        if num_vehicles < 1:
            return jsonify({'error': 'num_vehicles must be at least 1'}), 400
        
    except ValueError as e:
        return jsonify({
            'error': f'Invalid parameter type: {str(e)}. All numeric parameters must be integers.'
        }), 400
    
    # Prepare features
    try:
        features = prepare_features(
            hour=hour,
            day_of_week=day_of_week,
            month=month,
            year=year,
            day_of_month=day_of_month,
            num_vehicles=num_vehicles
        )
        
        # Make predictions
        rf_prediction = float(rf_model.predict(features)[0])
        lr_prediction = float(lr_model.predict(features)[0])
        
        # Calculate derived features for response
        is_weekend = (day_of_week >= 5)
        is_rush_hour = ((hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19))
        season = ((month % 12) // 3)
        season_names = ['Winter', 'Spring', 'Summer', 'Fall']
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        return jsonify({
            'success': True,
            'input': {
                'hour': hour,
                'day_of_week': day_of_week,
                'day_name': day_names[day_of_week],
                'month': month,
                'year': year,
                'day_of_month': day_of_month,
                'num_vehicles': num_vehicles,
                'is_weekend': is_weekend,
                'is_rush_hour': is_rush_hour,
                'season': season_names[season]
            },
            'predictions': {
                'random_forest': {
                    'predicted_injuries': round(rf_prediction, 4),
                    'model_r2_score': model_metadata['rf_test_r2'],
                    'model_mae': model_metadata['rf_test_mae']
                },
                'linear_regression': {
                    'predicted_injuries': round(lr_prediction, 4),
                    'model_r2_score': model_metadata['lr_test_r2'],
                    'model_mae': model_metadata['lr_test_mae']
                }
            },
            'recommendation': 'random_forest' if rf_prediction > lr_prediction else 'linear_regression'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Load models on startup
    print("Starting Flask API...")
    if load_models():
        print("API is ready!")
        print("\nAvailable endpoints:")
        print("  GET  / - API information")
        print("  GET  /health - Health check")
        print("  POST /predict - Make predictions")
        print("\nExample request:")
        print('  curl -X POST http://localhost:8080/predict \\')
        print('    -H "Content-Type: application/json" \\')
        print('    -d \'{"hour": 20, "day_of_week": 0, "month": 6}\'')
        print("\nStarting server on http://localhost:8080")
        app.run(debug=True, host='0.0.0.0', port=8080)
    else:
        print("Failed to load models. Please run main.py first.")
        exit(1)

