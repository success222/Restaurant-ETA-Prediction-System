# Restaurant ETA Prediction System

A machine learning project for predicting food delivery time using information about delivery distance, traffic, weather, vehicle type, order timing, and other delivery-related features.

## Project Overview

The goal of this project is to predict the time taken for a food delivery using regression models.

The target variable is:

`Time_taken(min)`

Three regression models were evaluated:

* Ridge Regression
* Random Forest Regressor
* Gradient Boosting Regressor

Random Forest achieved the best validation performance and was selected for further tuning.

## Dataset

The dataset contains information about food delivery orders, including:

* Delivery person age and rating
* Restaurant and delivery location coordinates
* Type of order
* Type of vehicle
* Weather conditions
* Road traffic density
* Festival information
* City
* Order and pickup times

The training dataset contains the target variable, while the provided test dataset does not contain `Time_taken(min)`.

## Data Preparation

### Cleaning

The data was cleaned by:

* Removing extra whitespace from column names and text values
* Converting relevant numeric columns to numeric types
* Cleaning the weather condition values
* Handling missing values through the preprocessing pipeline

### Feature Engineering

The following features were created:

* `distance_km` — distance between the restaurant and delivery location calculated using the Haversine formula
* `order_hour` — hour when the order was placed
* `pickup_hour` — hour when the order was picked up
* `order_to_pickup_minutes` — time between order placement and pickup

Some calculated distances were unrealistically large for food delivery. Distances above 100 km were therefore treated as missing values and handled by the imputation step.

## Preprocessing

A Scikit-learn preprocessing pipeline was used.

### Numerical Features

* Missing values: median imputation
* Scaling: `StandardScaler`

### Categorical Features

* Missing values: most frequent value
* Encoding: `OneHotEncoder`
* Unknown categories: ignored during transformation

Using a pipeline ensured that the same preprocessing steps were applied consistently during training and validation.

## Model Selection

The initial models were evaluated on a validation set.

| Model             |  MAE | RMSE |    R² |
| ----------------- | ---: | ---: | ----: |
| Ridge Regression  | 4.93 | 6.23 | 0.557 |
| Random Forest     | 3.20 | 4.03 | 0.815 |
| Gradient Boosting | 3.64 | 4.55 | 0.764 |

Random Forest produced the lowest MAE and RMSE and the highest R², so it was selected for further tuning.

## Model Tuning

`RandomizedSearchCV` with 3-fold cross-validation was used to tune the Random Forest.

The best parameters were:

```text
n_estimators = 200
max_depth = 20
min_samples_split = 2
min_samples_leaf = 2
max_features = 1.0
```

### Final Validation Results

| Metric |        Score |
| ------ | -----------: |
| MAE    | 3.19 minutes |
| RMSE   | 4.01 minutes |
| R²     |        0.817 |

The tuned model produced a small improvement over the initial Random Forest model.

## Error Analysis

The model performed reasonably well on average, but some individual predictions had substantially larger errors.

Some validation predictions were more than 15–20 minutes away from the actual delivery time. This indicates that although the model captured the general pattern in delivery times, it was not equally accurate for every individual delivery.

An actual-vs-predicted plot was also used to examine the model's predictions.

## Test Set

The provided test dataset does not contain the actual `Time_taken(min)` values.

Therefore, MAE, RMSE, and R² cannot be calculated for the test set. The trained model can still generate predictions for the test data, but those predictions cannot be evaluated without the actual target values.

## Conclusion

Random Forest achieved the best validation performance among the three models tested. After hyperparameter tuning, the final model achieved an MAE of 3.19 minutes, RMSE of 4.01 minutes, and R² of 0.817 on the validation set.
