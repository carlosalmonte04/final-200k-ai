# Project Scope

#### 1. **Predicting Injury Severity**

**Target**: `NUMBER OF PERSONS INJURED` (Binary: 0 = no injuries, 1+ = injuries occurred)
**Features**: Time of day, day of week, borough, weather data (external), vehicle types involved
**Why it works**: Temporal and spatial patterns affect crash severity
**Model**: Binary classification (Logistic Regression, Random Forest, XGBoost)

### External Data to Merge (For Better Predictions)

- Weather API data (temperature, precipitation, visibility at crash time)
- Traffic volume data (historical traffic patterns)
- Event calendars (holidays, special events that affect traffic)
- Economic indicators (unemployment rates, if available by borough)
- Infrastructure data (speed limits, presence of bike lanes)
