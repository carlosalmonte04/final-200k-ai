import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load the saved models and feature columns
print("Loading saved models...")
try:
    rf_model = joblib.load('rf_model.pkl')
    lr_model = joblib.load('lr_model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    model_metadata = joblib.load('model_metadata.pkl')
    print("Models loaded successfully!")
    print(f"Model performance: RF R²={model_metadata['rf_test_r2']:.4f}, LR R²={model_metadata['lr_test_r2']:.4f}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please run main.py first to train and save the models.")
    exit(1)

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
    
    # Add aggregate features (use default/mean values - these would ideally come from training data)
    # For now, using placeholder values
    features['AVG_INJURIES_BY_HOUR'] = [0.5]  # Should be loaded from training data
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
            df[col] = [0]  # Default to 0 for features not provided (borough dummies, vehicle types, etc.)
    
    # Ensure columns are in the correct order
    df = df[feature_columns]
    
    return df

# Example: Make a prediction
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Making Prediction Example")
    print("="*50)
    
    # Example 1: Simple prediction with minimal parameters
    sample_features = prepare_features(
        hour=20,
        day_of_week=0,  # Monday
        month=6,
        year=2025,
        day_of_year=180,
        week_of_year=32,
        is_weekend=False,
        is_rush_hour=True,
        is_night=False,
        is_morning=False,
        is_afternoon=False,
        is_evening=True,
        season=2
    )
    
    # Make predictions
    rf_prediction = rf_model.predict(sample_features)[0]
    lr_prediction = lr_model.predict(sample_features)[0]
    
    print(f"\nPredictions for Monday, 8 PM, June 2025 (Rush Hour):")
    print(f"Random Forest: {rf_prediction:.4f} persons injured")
    print(f"Linear Regression: {lr_prediction:.4f} persons injured")
    
    # Example 2: Weekend prediction
    print("\n" + "-"*50)
    weekend_features = prepare_features(
        hour=14,  # 2 PM
        day_of_week=5,  # Saturday
        month=7,
        is_weekend=True,
        is_rush_hour=False
    )
    
    rf_pred_weekend = rf_model.predict(weekend_features)[0]
    lr_pred_weekend = lr_model.predict(weekend_features)[0]
    
    print(f"Predictions for Saturday, 2 PM, July (Weekend):")
    print(f"Random Forest: {rf_pred_weekend:.4f} persons injured")
    print(f"Linear Regression: {lr_pred_weekend:.4f} persons injured")
    
    # Example 3: Minimal parameters (auto-calculated)
    print("\n" + "-"*50)
    simple_features = prepare_features(
        hour=9,
        day_of_week=2,  # Wednesday
        month=3
    )
    
    rf_pred_simple = rf_model.predict(simple_features)[0]
    lr_pred_simple = lr_model.predict(simple_features)[0]
    
    print(f"Predictions for Wednesday, 9 AM, March (auto-calculated features):")
    print(f"Random Forest: {rf_pred_simple:.4f} persons injured")
    print(f"Linear Regression: {lr_pred_simple:.4f} persons injured")

