# Project Scope

#### 1. **Predicting Injury Severity**

**Target**: `NUMBER OF PERSONS INJURED` (Binary: 0 = no injuries, 1+ = injuries occurred)
**Features**: Time of day, day of week, borough, weather data (external), vehicle types involved
**Why it works**: Temporal and spatial patterns affect crash severity
**Model**: Binary classification (Logistic Regression, Random Forest, XGBoost)

#### 2. **Predicting Pedestrian Fatalities**

**Target**: `NUMBER OF PEDESTRIANS KILLED` (Binary classification)
**Features**: Street type, time of day, contributing factors, weather data
**Why it works**: Driver behavior and environmental conditions correlate with pedestrian fatalities
**Model**: Classification models to identify high-risk scenarios

#### 3. **Predicting Crash Location Type**

**Target**: Predicting if crash occurs at intersections vs. mid-block
**Features**: `ON STREET NAME` vs `OFF STREET NAME`, latitude/longitude density
**Why it works**: Different crash dynamics at intersections vs. mid-block
**Model**: Classification problem with spatial features

#### 4. **Predicting Contributing Factors**

**Target**: Most common `CONTRIBUTING FACTOR` for a given location/time
**Features**: Borough, time, historical patterns, vehicle types
**Why it works**: Different areas/times have different failure modes
**Model**: Multi-class classification or text analysis

#### 5. **Predicting High-Risk Time Windows**

**Target**: Predict time periods with highest crash probability (regression)
**Features**: Time features (hour, day, month), borough, seasonality
**Why it works**: Temporal patterns exist in crash data
**Model**: Time series regression or clustering for risk periods

### Hidden Correlation Ideas (Butterfly Effect Approach)

#### 1. **Weather → Crash Severity**

**Question**: Does weather data (temperature, precipitation, visibility) correlate with crash severity better than we'd expect?

- Merge weather API data with crash timestamps
- Test if temperature alone predicts bicycle crashes better than vehicle factors alone

#### 2. **Day of Week → Injury Type**

**Question**: Do different days of the week have different injury patterns (pedestrian vs. motorist)?

- Find if weekend crashes have different injury distributions
- Test if weekday rush hours correlate more with cyclist injuries

#### 3. **Spatial Clustering → Contributing Factors**

**Question**: Do certain zip codes have different contributing factors (driver fatigue in residential areas vs. speeding on highways)?

- Cluster by location density
- Discover if neighborhood type predicts failure modes

#### 4. **Vehicle Type Combination → Severity**

**Question**: Do specific vehicle type combinations (e.g., car + bicycle vs. truck + pedestrian) have unexpectedly different fatality rates?

- Analyze if certain vehicle pairings are disproportionately dangerous

### External Data to Merge (For Better Predictions)

- Weather API data (temperature, precipitation, visibility at crash time)
- Traffic volume data (historical traffic patterns)
- Event calendars (holidays, special events that affect traffic)
- Economic indicators (unemployment rates, if available by borough)
- Infrastructure data (speed limits, presence of bike lanes)

### Why This Works

- ✅ **Real patterns exist**: Crash data has temporal, spatial, and seasonal patterns
- ✅ **Scientifically valid**: Traffic engineering and safety research uses this data
- ✅ **Practical value**: Insights can inform policy and safety improvements
- ✅ **Demonstrates ML skills**: Classification, regression, feature engineering, external data integration
