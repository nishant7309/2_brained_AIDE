# House Price Prediction Report

## Introduction

This report summarizes the development of a house price prediction model using machine learning techniques. The primary goal is to achieve a low Root Mean Squared Error (RMSE) on the test dataset. The approach involves data preprocessing, feature engineering, model selection, and ensemble methods.

## Preprocessing

### Data Loading and Combination

Training and test datasets were loaded using pandas. Identifiers were stored, and the target variable ('SalePrice') was separated from the training data. The datasets were then combined to ensure consistent preprocessing.

### Missing Value Handling

Missing values were imputed using `SimpleImputer` with different strategies:

-   Median imputation for 'LotFrontage' and 'GarageYrBlt'.
-   Constant imputation (0) for numerical features like 'GarageCars', 'BsmtFinSF1', etc.
-   Most frequent imputation for categorical features like 'MSZoning', 'Utilities', etc.

Columns 'Alley', 'PoolQC', 'Fence', and 'MiscFeature' were removed due to a high percentage of missing values.

### Outlier Handling

A log transformation was applied to the 'SalePrice' to reduce skewness.

### Feature Transformation

Ordinal encoding was used for features with inherent order (e.g., 'ExterQual') using a predefined mapping.

### Feature Engineering

Several new features were engineered:

-   Combined features: 'TotalSF', 'TotalBathrooms', 'TotalPorchSF'.
-   Age-related features: 'Age', 'RemodelAge', 'Age\_Garage'.
-   Interaction terms: 'OverallQual\_GrLivArea', 'TotalSF\_OverallQual', 'LotArea\_Neighborhood', 'YearBuilt\_OverallQual'.

### Data Splitting

The combined data was split back into training and test sets. The training set was further split into training and validation sets.

### Feature Scaling and Encoding

Numerical features were scaled using `RobustScaler`. Categorical features were one-hot encoded using `OneHotEncoder`.

### Feature Selection

`VarianceThreshold` was applied to remove features with low variance.

## Modeling Methods

### Model

XGBoost Regressor was selected as a high-performing model for this task.

### Training

The XGBoost model was trained with the following hyperparameters:

-   `objective`='reg:squarederror'
-   `n_estimators`=1000
-   `learning_rate`=0.05
-   `max_depth`=5
-   `min_child_weight`=1
-   `gamma`=0
-   `subsample`=0.8
-   `colsample_bytree`=0.8
-   `reg_alpha`=0.005
-   `random_state`=42
-   `n_jobs`=-1
-   `early_stopping_rounds`=50

The model was trained with early stopping, monitoring the validation set for performance improvement.

## Results Discussion

The XGBoost model achieved a validation RMSE of 0.1376. Post-training, predictions were made on the test data and inverse-transformed to the original scale.

## Future Work

-   Hyperparameter tuning using Bayesian optimization or Optuna.
-   Experiment with other feature selection techniques (e.g., RFE, SelectFromModel).
-   Explore additional feature engineering opportunities.
-   Consider more advanced ensemble methods, such as stacking with multiple base models (LightGBM, CatBoost, Neural Networks).
-   Address potential feature interactions using automated feature interaction search.
