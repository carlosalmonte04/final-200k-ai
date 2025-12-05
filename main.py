import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
# Load the CSV file
df = pd.read_csv('Motor_Vehicle_Collisions_-_Crashes.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Check for missing values in key columns
print(f"\nMissing values in key columns:")
print(df[['CRASH DATE', 'CRASH TIME', 'NUMBER OF PERSONS INJURED']].isnull().sum())

# Clean the data - remove rows with missing crash time or date
df = df.dropna(subset=['CRASH DATE', 'CRASH TIME', 'NUMBER OF PERSONS INJURED'])
print(f"\nShape after removing missing values: {df.shape}")

# Convert crash date and time to datetime
print("\nProcessing date and time features...")
df['CRASH DATETIME'] = pd.to_datetime(
    df['CRASH DATE'].astype(str) + ' ' + df['CRASH TIME'].astype(str),
    errors='coerce'
)

# Remove rows where datetime conversion failed
df = df.dropna(subset=['CRASH DATETIME'])
print(f"Shape after datetime conversion: {df.shape}")

# Extract time-based features
df['HOUR'] = df['CRASH DATETIME'].dt.hour
df['DAY_OF_WEEK'] = df['CRASH DATETIME'].dt.dayofweek  # 0=Monday, 6=Sunday
df['MONTH'] = df['CRASH DATETIME'].dt.month
df['DAY_OF_MONTH'] = df['CRASH DATETIME'].dt.day
df['YEAR'] = df['CRASH DATETIME'].dt.year
df['DAY_OF_YEAR'] = df['CRASH DATETIME'].dt.dayofyear
df['WEEK_OF_YEAR'] = df['CRASH DATETIME'].dt.isocalendar().week

# Binary time features
df['IS_WEEKEND'] = (df['DAY_OF_WEEK'] >= 5).astype(int)
df['IS_RUSH_HOUR'] = ((df['HOUR'] >= 7) & (df['HOUR'] <= 9) | 
                      (df['HOUR'] >= 17) & (df['HOUR'] <= 19)).astype(int)
df['IS_NIGHT'] = ((df['HOUR'] >= 22) | (df['HOUR'] <= 5)).astype(int)
df['IS_MORNING'] = ((df['HOUR'] >= 6) & (df['HOUR'] <= 11)).astype(int)
df['IS_AFTERNOON'] = ((df['HOUR'] >= 12) & (df['HOUR'] <= 17)).astype(int)
df['IS_EVENING'] = ((df['HOUR'] >= 18) & (df['HOUR'] <= 21)).astype(int)

# Season feature (0=Winter, 1=Spring, 2=Summer, 3=Fall)
df['SEASON'] = ((df['MONTH'] % 12) // 3).astype(int)

# Create cyclical features for time (sine/cosine encoding)
df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
df['DAY_OF_WEEK_SIN'] = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)
df['DAY_OF_WEEK_COS'] = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)
df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
df['DAY_OF_MONTH_SIN'] = np.sin(2 * np.pi * df['DAY_OF_MONTH'] / 31)
df['DAY_OF_MONTH_COS'] = np.cos(2 * np.pi * df['DAY_OF_MONTH'] / 31)
df['DAY_OF_YEAR_SIN'] = np.sin(2 * np.pi * df['DAY_OF_YEAR'] / 365)
df['DAY_OF_YEAR_COS'] = np.cos(2 * np.pi * df['DAY_OF_YEAR'] / 365)
df['WEEK_OF_YEAR_SIN'] = np.sin(2 * np.pi * df['WEEK_OF_YEAR'] / 52)
df['WEEK_OF_YEAR_COS'] = np.cos(2 * np.pi * df['WEEK_OF_YEAR'] / 52)

# Interaction features
df['RUSH_HOUR_WEEKEND'] = df['IS_RUSH_HOUR'] * df['IS_WEEKEND']
df['NIGHT_WEEKEND'] = df['IS_NIGHT'] * df['IS_WEEKEND']
df['MORNING_WEEKEND'] = df['IS_MORNING'] * df['IS_WEEKEND']

print("\n" + "="*50)
print("Feature Engineering...")
print("="*50)

# Prepare base feature columns
feature_columns = [
    'HOUR', 'DAY_OF_WEEK', 'MONTH', 'YEAR', 'DAY_OF_YEAR', 'WEEK_OF_YEAR',
    'IS_WEEKEND', 'IS_RUSH_HOUR', 'IS_NIGHT', 'IS_MORNING', 'IS_AFTERNOON', 'IS_EVENING',
    'SEASON',
    'HOUR_SIN', 'HOUR_COS', 'DAY_OF_WEEK_SIN', 'DAY_OF_WEEK_COS',
    'MONTH_SIN', 'MONTH_COS', 'DAY_OF_MONTH_SIN', 'DAY_OF_MONTH_COS',
    'DAY_OF_YEAR_SIN', 'DAY_OF_YEAR_COS', 'WEEK_OF_YEAR_SIN', 'WEEK_OF_YEAR_COS',
    'RUSH_HOUR_WEEKEND', 'NIGHT_WEEKEND', 'MORNING_WEEKEND'
]

# Add borough if available (one-hot encode)
if 'BOROUGH' in df.columns:
    print("Processing BOROUGH feature...")
    df['BOROUGH'] = df['BOROUGH'].fillna('UNKNOWN')
    borough_dummies = pd.get_dummies(df['BOROUGH'], prefix='BOROUGH')
    feature_columns.extend(borough_dummies.columns.tolist())
    df = pd.concat([df, borough_dummies], axis=1)

# Process ZIP CODE if available
if 'ZIP CODE' in df.columns:
    print("Processing ZIP CODE feature...")
    df['ZIP CODE'] = df['ZIP CODE'].fillna('UNKNOWN').astype(str)
    # Only keep top ZIP codes to avoid too many features
    top_zip_codes = df['ZIP CODE'].value_counts().head(20).index
    for zip_code in top_zip_codes:
        df[f'ZIP_{zip_code}'] = (df['ZIP CODE'] == zip_code).astype(int)
        feature_columns.append(f'ZIP_{zip_code}')

# Process vehicle type codes (typically VEHICLE TYPE CODE 1, 2, 3, etc.)
print("Processing VEHICLE TYPE features...")
vehicle_type_columns = [col for col in df.columns if 'VEHICLE TYPE CODE' in col.upper()]
if vehicle_type_columns:
    all_vehicle_types = set()
    for col in vehicle_type_columns:
        all_vehicle_types.update(df[col].dropna().unique())
    
    # Keep only top vehicle types
    vehicle_type_counts = {}
    for col in vehicle_type_columns:
        for vtype in df[col].dropna():
            vehicle_type_counts[vtype] = vehicle_type_counts.get(vtype, 0) + 1
    
    top_vehicle_types = sorted(vehicle_type_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    top_vehicle_types = [vtype for vtype, count in top_vehicle_types if pd.notna(vtype) and str(vtype).strip() != '']
    
    for vtype in top_vehicle_types:
        vtype_clean = str(vtype).replace(' ', '_').replace('/', '_').replace('-', '_')[:30]
        df[f'VEHICLE_TYPE_{vtype_clean}'] = 0
        for col in vehicle_type_columns:
            df[f'VEHICLE_TYPE_{vtype_clean}'] += (df[col] == vtype).astype(int)
        df[f'VEHICLE_TYPE_{vtype_clean}'] = (df[f'VEHICLE_TYPE_{vtype_clean}'] > 0).astype(int)
        feature_columns.append(f'VEHICLE_TYPE_{vtype_clean}')

# Process contributing factors (typically CONTRIBUTING FACTOR VEHICLE 1, 2, etc.)
print("Processing CONTRIBUTING FACTOR features...")
contributing_factor_columns = [col for col in df.columns if 'CONTRIBUTING FACTOR' in col.upper()]
if contributing_factor_columns:
    factor_counts = {}
    for col in contributing_factor_columns:
        for factor in df[col].dropna():
            if pd.notna(factor) and str(factor).strip() not in ['', 'Unspecified', 'Unspecified']:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1
    
    top_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    top_factors = [factor for factor, count in top_factors]
    
    for factor in top_factors:
        factor_clean = str(factor).replace(' ', '_').replace('/', '_').replace('-', '_')[:30]
        df[f'FACTOR_{factor_clean}'] = 0
        for col in contributing_factor_columns:
            df[f'FACTOR_{factor_clean}'] += (df[col] == factor).astype(int)
        df[f'FACTOR_{factor_clean}'] = (df[f'FACTOR_{factor_clean}'] > 0).astype(int)
        feature_columns.append(f'FACTOR_{factor_clean}')

# Process number of vehicles involved
print("Processing vehicle count features...")
vehicle_count_cols = [col for col in df.columns if 'VEHICLE TYPE CODE' in col.upper()]
if vehicle_count_cols:
    df['NUM_VEHICLES'] = 0
    for col in vehicle_count_cols:
        df['NUM_VEHICLES'] += df[col].notna().astype(int)
    feature_columns.append('NUM_VEHICLES')

# Process other injury/killed counts as potential features
if 'NUMBER OF PERSONS KILLED' in df.columns:
    df['HAS_FATALITIES'] = (df['NUMBER OF PERSONS KILLED'] > 0).astype(int)
    feature_columns.append('HAS_FATALITIES')

if 'NUMBER OF PEDESTRIANS INJURED' in df.columns:
    df['HAS_PEDESTRIAN_INJURIES'] = (df['NUMBER OF PEDESTRIANS INJURED'] > 0).astype(int)
    # feature_columns.append('HAS_PEDESTRIAN_INJURIES')

if 'NUMBER OF CYCLIST INJURED' in df.columns:
    df['HAS_CYCLIST_INJURIES'] = (df['NUMBER OF CYCLIST INJURED'] > 0).astype(int)
    # feature_columns.append('HAS_CYCLIST_INJURIES')

if 'NUMBER OF MOTORIST INJURED' in df.columns:
    df['HAS_MOTORIST_INJURIES'] = (df['NUMBER OF MOTORIST INJURED'] > 0).astype(int)
    # feature_columns.append('HAS_MOTORIST_INJURIES')

# Process street name features (extract street type)
print("Processing street name features...")
if 'ON STREET NAME' in df.columns:
    df['ON_STREET_TYPE'] = df['ON STREET NAME'].fillna('').astype(str).str.extract(r'\b(AVE|ST|BLVD|RD|DR|PL|PKWY|LN|CT|WAY|BLVD)\b', expand=False)
    df['ON_STREET_TYPE'] = df['ON_STREET_TYPE'].fillna('OTHER')
    top_street_types = df['ON_STREET_TYPE'].value_counts().head(10).index
    for stype in top_street_types:
        df[f'STREET_TYPE_{stype}'] = (df['ON_STREET_TYPE'] == stype).astype(int)
        feature_columns.append(f'STREET_TYPE_{stype}')

# Create aggregate features (average injuries by various groupings)
print("Creating aggregate features...")
if 'BOROUGH' in df.columns:
    df['AVG_INJURIES_BY_BOROUGH'] = df.groupby('BOROUGH')['NUMBER OF PERSONS INJURED'].transform('mean')
    feature_columns.append('AVG_INJURIES_BY_BOROUGH')

df['AVG_INJURIES_BY_HOUR'] = df.groupby('HOUR')['NUMBER OF PERSONS INJURED'].transform('mean')
feature_columns.append('AVG_INJURIES_BY_HOUR')

df['AVG_INJURIES_BY_DAY_OF_WEEK'] = df.groupby('DAY_OF_WEEK')['NUMBER OF PERSONS INJURED'].transform('mean')
feature_columns.append('AVG_INJURIES_BY_DAY_OF_WEEK')

df['AVG_INJURIES_BY_MONTH'] = df.groupby('MONTH')['NUMBER OF PERSONS INJURED'].transform('mean')
feature_columns.append('AVG_INJURIES_BY_MONTH')

# Process latitude/longitude if available (create location clusters)
if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
    print("Processing location features...")
    # Filter valid coordinates
    valid_coords = df['LATITUDE'].between(40.4, 40.9) & df['LONGITUDE'].between(-74.3, -73.7) & df['LATITUDE'].notna() & df['LONGITUDE'].notna()
    
    # Create location bins (only for valid coordinates)
    df['LAT_BIN'] = np.nan
    df['LON_BIN'] = np.nan
    
    if valid_coords.sum() > 0:
        df.loc[valid_coords, 'LAT_BIN'] = pd.cut(df.loc[valid_coords, 'LATITUDE'], bins=10, labels=False, duplicates='drop')
        df.loc[valid_coords, 'LON_BIN'] = pd.cut(df.loc[valid_coords, 'LONGITUDE'], bins=10, labels=False, duplicates='drop')
    
    df['LOCATION_CLUSTER'] = df['LAT_BIN'].astype(str) + '_' + df['LON_BIN'].astype(str)
    df['LOCATION_CLUSTER'] = df['LOCATION_CLUSTER'].replace('nan_nan', 'UNKNOWN')
    
    # Only keep top location clusters
    valid_clusters = df[df['LOCATION_CLUSTER'] != 'UNKNOWN']['LOCATION_CLUSTER']
    if len(valid_clusters) > 0:
        top_clusters = valid_clusters.value_counts().head(20).index
        for cluster in top_clusters:
            cluster_clean = str(cluster).replace(' ', '_').replace('.', '_')
            df[f'LOC_{cluster_clean}'] = (df['LOCATION_CLUSTER'] == cluster).astype(int)
            feature_columns.append(f'LOC_{cluster_clean}')

print(f"Total features created: {len(feature_columns)}")

X = df[feature_columns].fillna(0)
y = df['NUMBER OF PERSONS INJURED'].astype(float)

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nTarget statistics:")
print(y.describe())

# Split the data
print("\nSplitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train models
print("\n" + "="*50)
print("Hyperparameter Tuning for Random Forest...")
print("="*50)

# For large datasets, use a sample for faster hyperparameter tuning
# This speeds up the process while still finding good parameters
SAMPLE_SIZE_FOR_TUNING = 50000  # Use 50k samples for tuning
if len(X_train) > SAMPLE_SIZE_FOR_TUNING:
    print(f"Using {SAMPLE_SIZE_FOR_TUNING:,} samples for hyperparameter tuning (out of {len(X_train):,} total)")
    rng = np.random.default_rng(42)  # For reproducibility
    tuning_indices = rng.choice(len(X_train), SAMPLE_SIZE_FOR_TUNING, replace=False)
    X_train_tuning = X_train.iloc[tuning_indices]
    y_train_tuning = y_train.iloc[tuning_indices]
else:
    X_train_tuning = X_train
    y_train_tuning = y_train

# Define parameter grid for hyperparameter tuning
param_distributions = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [8, 10, 12, 15, None],
    'min_samples_split': [10, 20, 30, 50],
    'min_samples_leaf': [5, 10, 15, 20],
    'max_features': ['sqrt', 'log2', 0.5, 0.7]
}

# Create base model
rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

# Perform randomized search with cross-validation
# Using fewer iterations (n_iter=20) to balance time vs. thoroughness
# Using 3-fold CV to speed up the process
print("Performing randomized search with 3-fold cross-validation...")
print("This may take a few minutes...")

rf_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_distributions,
    n_iter=30,  # Number of parameter settings sampled
    cv=3,  # 3-fold cross-validation
    scoring='neg_mean_squared_error',  # We want to minimize MSE
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf_search.fit(X_train_tuning, y_train_tuning)

# Get best parameters
best_params = rf_search.best_params_
print(f"\nBest hyperparameters found:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print(f"Best CV score (negative MSE): {rf_search.best_score_:.4f}")

# Train final model with best parameters
print("\n" + "="*50)
print("Training Random Forest with Best Hyperparameters...")
print("="*50)

rf_model = RandomForestRegressor(
    **best_params,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Make predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Evaluate Random Forest
rf_train_mse = mean_squared_error(y_train, y_train_pred)
rf_test_mse = mean_squared_error(y_test, y_test_pred)
rf_train_mae = mean_absolute_error(y_train, y_train_pred)
rf_test_mae = mean_absolute_error(y_test, y_test_pred)
rf_train_r2 = r2_score(y_train, y_train_pred)
rf_test_r2 = r2_score(y_test, y_test_pred)

print(f"\nRandom Forest Results:")
print(f"Train MSE: {rf_train_mse:.4f}")
print(f"Test MSE: {rf_test_mse:.4f}")
print(f"Train MAE: {rf_train_mae:.4f}")
print(f"Test MAE: {rf_test_mae:.4f}")
print(f"Train R²: {rf_train_r2:.4f}")
print(f"Test R²: {rf_test_r2:.4f}")
print("\n" + "="*50)
print("Hyperparameter Tuning for Linear Regression...")
print("="*50)

# Define parameter grid for regularized linear models
# We'll test Ridge, Lasso, and ElasticNet with different alpha values
lr_param_distributions = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # Only used for ElasticNet
}

# Test different linear models
print("Testing Ridge, Lasso, and ElasticNet regression...")

# Ridge Regression
ridge_base = Ridge(random_state=42, max_iter=1000)
ridge_search = RandomizedSearchCV(
    estimator=ridge_base,
    param_distributions={'alpha': lr_param_distributions['alpha']},
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=0
)
ridge_search.fit(X_train_tuning, y_train_tuning)
best_ridge_params = ridge_search.best_params_
best_ridge_score = ridge_search.best_score_

# Lasso Regression
lasso_base = Lasso(random_state=42, max_iter=1000)
lasso_search = RandomizedSearchCV(
    estimator=lasso_base,
    param_distributions={'alpha': lr_param_distributions['alpha']},
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=0
)
lasso_search.fit(X_train_tuning, y_train_tuning)
best_lasso_params = lasso_search.best_params_
best_lasso_score = lasso_search.best_score_

# ElasticNet Regression
elasticnet_base = ElasticNet(random_state=42, max_iter=1000)
elasticnet_search = RandomizedSearchCV(
    estimator=elasticnet_base,
    param_distributions=lr_param_distributions,
    n_iter=20,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=0
)
elasticnet_search.fit(X_train_tuning, y_train_tuning)
best_elasticnet_params = elasticnet_search.best_params_
best_elasticnet_score = elasticnet_search.best_score_

# Compare all models and select the best one
models_comparison = {
    'Ridge': (best_ridge_score, best_ridge_params),
    'Lasso': (best_lasso_score, best_lasso_params),
    'ElasticNet': (best_elasticnet_score, best_elasticnet_params)
}

best_lr_model_name = max(models_comparison, key=lambda x: models_comparison[x][0])
best_lr_score, best_lr_params = models_comparison[best_lr_model_name]

print(f"\nBest Linear Regression model: {best_lr_model_name}")
print(f"Best parameters: {best_lr_params}")
print(f"Best CV score (negative MSE): {best_lr_score:.4f}")

# Train final linear regression model with best parameters
print("\n" + "="*50)
print(f"Training {best_lr_model_name} with Best Hyperparameters...")
print("="*50)

if best_lr_model_name == 'Ridge':
    lr_model = Ridge(**best_lr_params, random_state=42, max_iter=1000)
elif best_lr_model_name == 'Lasso':
    lr_model = Lasso(**best_lr_params, random_state=42, max_iter=1000)
else:  # ElasticNet
    lr_model = ElasticNet(**best_lr_params, random_state=42, max_iter=1000)

lr_model.fit(X_train, y_train)

y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)

lr_train_mse = mean_squared_error(y_train, y_train_pred_lr)
lr_test_mse = mean_squared_error(y_test, y_test_pred_lr)
lr_train_mae = mean_absolute_error(y_train, y_train_pred_lr)
lr_test_mae = mean_absolute_error(y_test, y_test_pred_lr)
lr_train_r2 = r2_score(y_train, y_train_pred_lr)
lr_test_r2 = r2_score(y_test, y_test_pred_lr)
print(f"\nLinear Regression Results:")
print(f"Train MSE: {lr_train_mse:.4f}")
print(f"Test MSE: {lr_test_mse:.4f}")
print(f"Train MAE: {lr_train_mae:.4f}")
print(f"Test MAE: {lr_test_mae:.4f}")
print(f"Train R²: {lr_train_r2:.4f}")
print(f"Test R²: {lr_test_r2:.4f}")
# Feature importance (Random Forest)
print("\n" + "="*50)
print("Top 20 Most Important Features:")
print("="*50)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(20))

# Visualizations
print("\n" + "="*50)
print("Creating visualizations...")
print("="*50)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# 1. Actual vs Predicted (Random Forest)
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, s=1)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Persons Injured')
axes[0, 0].set_ylabel('Predicted Persons Injured')
axes[0, 0].set_title('Random Forest: Actual vs Predicted')
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals plot (Random Forest)
residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5, s=1)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Persons Injured')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Random Forest: Residuals Plot')
axes[0, 1].grid(True, alpha=0.3)

# 3. Feature importance
top_features = feature_importance.head(10)
axes[1, 0].barh(top_features['feature'], top_features['importance'])
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 10 Feature Importances')
axes[1, 0].invert_yaxis()

# 4. Injuries by hour of day
hourly_injuries = df.groupby('HOUR')['NUMBER OF PERSONS INJURED'].mean()
axes[1, 1].plot(hourly_injuries.index, hourly_injuries.values, marker='o')
axes[1, 1].set_xlabel('Hour of Day')
axes[1, 1].set_ylabel('Average Persons Injured')
axes[1, 1].set_title('Average Injuries by Hour of Day')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks(range(0, 24, 2))

# 5. R² Score comparison between models
axes[2, 0].bar(['Random Forest', 'Linear Regression'], [rf_test_r2, lr_test_r2])
axes[2, 0].set_ylabel('R² Score')
axes[2, 0].set_title('R² Score Comparison (Test Set)')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

# 6. MAE comparison between models
axes[2, 1].bar(['Random Forest', 'Linear Regression'], [rf_test_mae, lr_test_mae])
axes[2, 1].set_ylabel('MAE (Mean Absolute Error)')
axes[2, 1].set_title('MAE Comparison (Test Set)')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
print("Visualizations saved to 'model_results.png'")

# Summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Best model: Random Forest Regressor")
print(f"Test R² Score: {rf_test_r2:.4f}")
print(f"Test MAE: {rf_test_mae:.4f} (average error in number of persons injured)")
print(f"\nThe model uses {len(feature_columns)} features including:")
print(f"- Temporal features (hour, day, month, season, year)")
print(f"- Time period features (rush hour, night, morning, afternoon, evening)")
print(f"- Cyclical time encoding (sine/cosine)")
print(f"- Interaction features (rush hour × weekend, etc.)")
print(f"- Location features (borough, ZIP code, street types, location clusters)")
print(f"- Vehicle features (vehicle types, number of vehicles)")
print(f"- Contributing factors")
print(f"- Aggregate statistics (average injuries by location/time)")

# Save the trained models
print("\n" + "="*50)
print("Saving trained models...")
print("="*50)

# Create timestamp for versioning (optional)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save Random Forest model
rf_model_filename = f'rf_model_{timestamp}.pkl'
joblib.dump(rf_model, rf_model_filename)
print(f"Random Forest model saved to '{rf_model_filename}'")

# Also save without timestamp for easy loading
joblib.dump(rf_model, 'rf_model.pkl')
print("Random Forest model also saved to 'rf_model.pkl' (latest version)")

# Save Linear Regression model
lr_model_filename = f'lr_model_{timestamp}.pkl'
joblib.dump(lr_model, lr_model_filename)
print(f"Linear Regression model saved to '{lr_model_filename}'")

# Also save without timestamp for easy loading
joblib.dump(lr_model, 'lr_model.pkl')
print("Linear Regression model also saved to 'lr_model.pkl' (latest version)")

# Save feature columns (critical for making predictions later)
joblib.dump(feature_columns, 'feature_columns.pkl')
print("Feature columns saved to 'feature_columns.pkl'")

# Save model metadata
model_metadata = {
    'rf_test_r2': rf_test_r2,
    'rf_test_mae': rf_test_mae,
    'rf_test_mse': rf_test_mse,
    'lr_test_r2': lr_test_r2,
    'lr_test_mae': lr_test_mae,
    'lr_test_mse': lr_test_mse,
    'num_features': len(feature_columns),
    'timestamp': timestamp
}
joblib.dump(model_metadata, 'model_metadata.pkl')
print("Model metadata saved to 'model_metadata.pkl'")
print("\nAll models saved successfully!")

# Make predictions on sample data
# Create a single-row DataFrame with all required features
sample_data = {}

# Base temporal features
sample_data['HOUR'] = [10]
sample_data['DAY_OF_WEEK'] = [0]
sample_data['MONTH'] = [1]
sample_data['YEAR'] = [2025]
sample_data['DAY_OF_YEAR'] = [180]
sample_data['WEEK_OF_YEAR'] = [32]
sample_data['IS_WEEKEND'] = [0]
sample_data['IS_RUSH_HOUR'] = [1]
sample_data['IS_NIGHT'] = [0]
sample_data['IS_MORNING'] = [1]
sample_data['IS_AFTERNOON'] = [0]
sample_data['IS_EVENING'] = [0]
sample_data['SEASON'] = [2]

# Cyclical features
sample_data['HOUR_SIN'] = [np.sin(2 * np.pi * 20 / 24)]
sample_data['HOUR_COS'] = [np.cos(2 * np.pi * 20 / 24)]
sample_data['DAY_OF_WEEK_SIN'] = [np.sin(2 * np.pi * 0 / 7)]
sample_data['DAY_OF_WEEK_COS'] = [np.cos(2 * np.pi * 0 / 7)]
sample_data['MONTH_SIN'] = [np.sin(2 * np.pi * 6 / 12)]
sample_data['MONTH_COS'] = [np.cos(2 * np.pi * 6 / 12)]
sample_data['DAY_OF_MONTH_SIN'] = [np.sin(2 * np.pi * 15 / 31)]
sample_data['DAY_OF_MONTH_COS'] = [np.cos(2 * np.pi * 15 / 31)]
sample_data['DAY_OF_YEAR_SIN'] = [np.sin(2 * np.pi * 180 / 365)]
sample_data['DAY_OF_YEAR_COS'] = [np.cos(2 * np.pi * 180 / 365)]
sample_data['WEEK_OF_YEAR_SIN'] = [np.sin(2 * np.pi * 32 / 52)]
sample_data['WEEK_OF_YEAR_COS'] = [np.cos(2 * np.pi * 32 / 52)]

# Interaction features
sample_data['RUSH_HOUR_WEEKEND'] = [0]
sample_data['NIGHT_WEEKEND'] = [0]
sample_data['MORNING_WEEKEND'] = [0]

# Aggregate features - use mean values from training data
sample_data['AVG_INJURIES_BY_BOROUGH'] = [df['AVG_INJURIES_BY_BOROUGH'].mean()] if 'AVG_INJURIES_BY_BOROUGH' in df.columns else [0]
sample_data['AVG_INJURIES_BY_HOUR'] = [df['AVG_INJURIES_BY_HOUR'].mean()]
sample_data['AVG_INJURIES_BY_DAY_OF_WEEK'] = [df['AVG_INJURIES_BY_DAY_OF_WEEK'].mean()]
sample_data['AVG_INJURIES_BY_MONTH'] = [df['AVG_INJURIES_BY_MONTH'].mean()]

# Create DataFrame with only the base features
sample_df = pd.DataFrame(sample_data)

# Add all other features from feature_columns, setting them to 0
for feature in feature_columns:
    if feature not in sample_df.columns:
        sample_df[feature] = [0]

# Reorder columns to match feature_columns order
sample_df = sample_df[feature_columns]

# Make prediction
prediction = rf_model.predict(sample_df)
print(f"Prediction for sample data: {prediction[0]:.4f} persons injured")

