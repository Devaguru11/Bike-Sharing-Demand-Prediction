# 🚴‍♂️ Bike Sharing Demand Prediction Project

## 📘 Overview
This project aims to **predict the number of bikes rented** in a bike-sharing system using historical data.  
The prediction is based on **environmental** and **temporal** factors such as temperature, humidity, windspeed, and time of day.

Accurate demand prediction helps:
- Ensure enough bikes are available during peak hours.
- Reduce maintenance and idle costs.
- Improve customer satisfaction and resource management.

---

## 📊 Dataset Description

**Source:** [Kaggle - Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand)

| Feature | Description |
|----------|--------------|
| `datetime` | Date and time of the record |
| `temp` | Temperature in Celsius |
| `humidity` | Humidity level |
| `windspeed` | Wind speed |
| `count` | Number of bikes rented (target variable) |

**Train Data Size:** 10,886 rows  
**Test Data Size:** 6,500 rows

---

## ⚙️ Approach

### Step-by-Step Methodology

#### 1️⃣ Data Loading
- Imported train and test datasets using **pandas**.

#### 2️⃣ Feature Engineering
- Extracted time-based features from `datetime`:  
  `hour`, `day`, `month`, `year`, `dayofweek`, and `weekend`.
- These features capture **seasonal**, **daily**, and **weekly** usage patterns.

## 3️⃣ Feature Selection
Selected relevant features for the model:

['temp', 'humidity', 'windspeed', 'hour', 'day', 'month', 'year', 'dayofweek', 'weekend']

##4️⃣ Data Transformation
Applied log transformation to the target variable count to reduce skewness and stabilize variance.

##5️⃣ Model Building
Used Random Forest Regressor from scikit-learn for its strong performance on tabular data and ability to handle non-linear relationships.

## 6️⃣ Model Tuning
Optimized model parameters:

#python
Copy code
n_estimators = 300
max_depth = 20
min_samples_split = 5
min_samples_leaf = 2
##7️⃣ Evaluation Metric
Chose RMSLE (Root Mean Squared Log Error) as the evaluation metric.

RMSLE penalizes large relative errors and suits count-based predictions.

🧠 Code Review & Visualization
🔑 Key Code Snippets
Feature Extraction:

python
Copy code
train['hour'] = train['datetime'].dt.hour
train['month'] = train['datetime'].dt.month
train['dayofweek'] = train['datetime'].dt.dayofweek
train['weekend'] = train['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
Model Training:

python
Copy code
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
Prediction and Evaluation:

python
Copy code
y_pred_log = model.predict(X_valid)
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_log)))
print("Validation RMSLE:", rmsle)

## 📉 Visualizations
Actual vs Predicted Scatter Plot

Shows how closely the predictions align with real data.

Indicates a strong correlation between actual and predicted values.

(Insert scatter plot image here)

Feature Importance (Optional)

Visualize which features influence demand the most (e.g., hour, temperature, humidity).

(Insert feature importance plot here)

## 📈 Results & Submission
🧮 Model Performance
Metric	Value
Validation RMSLE	0.33
Training Time	~15 seconds
Model Used	Random Forest Regressor

##🔍 Findings
Hour, temperature, and humidity were the most influential features.

Peak demand occurs during morning and evening hours.

Weather and weekends significantly affect rental behavior.

## 💾 Output File
The final predictions are stored in:

Copy code
submission_optimized.csv
Columns:

datetime

count (predicted bike rental count)

## 🧾 Conclusion
# ✅ Summary
This project successfully implemented a machine learning regression model to forecast bike rental demand.
The optimized Random Forest Regressor achieved a low RMSLE of 0.33, effectively capturing real-world demand patterns.

##💡 Key Learnings
Feature engineering greatly improves model accuracy.

Log transformation helps handle skewed count data.

RMSLE is ideal for evaluating demand prediction tasks with large value ranges.

## 🚀 Future Improvements
Add more features like holiday, season, and weather.

Try advanced models such as XGBoost, LightGBM, or CatBoost.

Use cross-validation for more robust performance evaluation.


