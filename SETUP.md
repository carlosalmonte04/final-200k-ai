# Setup Instructions

## Virtual Environment Setup

The virtual environment has been created and packages installed. To activate it:

```bash
# Activate virtual environment
source venv/bin/activate

# On Windows (if using Git Bash):
# source venv/Scripts/activate
```

## Running the Model

Once the virtual environment is activated, run:

```bash
python main.py
```

This will:

1. Load the crash data CSV
2. Preprocess date/time features
3. Train Random Forest and Linear Regression models
4. Evaluate model performance
5. Generate visualizations saved as `model_results.png`

## Model Details

The model predicts `NUMBER OF PERSONS INJURED` based on:

- **Hour of day** (0-23)
- **Day of week** (0=Monday, 6=Sunday)
- **Month** (1-12)
- **Day of month** (1-31)
- **Is weekend** (binary)
- **Is rush hour** (binary: 7-9am or 5-7pm)
- **Cyclical time features** (sine/cosine encoding for better temporal patterns)
- **Borough** (one-hot encoded, if available)

## Expected Output

The script will display:

- Dataset statistics
- Model performance metrics (MSE, MAE, R²)
- Feature importance rankings
- Visualizations comparing actual vs predicted values

## Deactivating Virtual Environment

When done, deactivate the environment:

```bash
deactivate
```
