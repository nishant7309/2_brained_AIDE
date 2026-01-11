```markdown
## Technical Report: House Price Prediction via Stacked Ensemble

### Introduction
This report details the development of a stacked ensemble model for predicting house prices, targeting minimization of the Root Mean Squared Error (RMSE). The approach combines data preprocessing, feature engineering, and a two-level stacking architecture leveraging LightGBM, XGBoost, CatBoost, and Ridge Regression.

### Preprocessing
1.  **Data Loading and Combination:** Training and testing data were loaded using pandas, combined for consistent preprocessing, and the target variable (`SalePrice`) log-transformed.
2.  **Outlier Handling:** Two outliers with `GrLivArea > 4000` were removed from the training data.
3.  **Missing Value Imputation:** Missing values were imputed based on feature context. Categorical features were filled with "None" or mode, while numerical features were filled with 0 or the median `LotFrontage` within each neighborhood. The `Utilities` column was dropped.
4.  **Feature Engineering:** New features were created, including `TotalSF`, `TotalBath`, `TotalPorchSF`, and binary indicators (`HasPool`, `Has2ndfloor`, `HasGarage`, `HasBsmt`, `HasFireplace`). Categorical features disguised as numbers were converted to strings.
5.  **Feature Transformation:** Ordinal features were mapped to numerical values. Skewed numerical features were corrected using Box-Cox transformation. Finally, categorical features were one-hot encoded.

### Modelling Methods
1.  **Level 0 (Base Models):**
    *   **LightGBM:** Gradient boosting framework with leaf-wise growth.
    *   **XGBoost:** Regularized gradient boosting framework with level-wise tree building.
    *   **CatBoost:** Gradient boosting library designed for categorical data.
    *   **Ridge Regression:** Linear model with L2 regularization.
2.  **Level 1 (Meta-Model):**
    *   **Ridge Regression:** Regularized linear model to combine Level 0 predictions.

### Results Discussion
1.  **Validation Framework:** 10-fold cross-validation was used.
2.  **Training and Prediction:** Level 0 models were trained within each fold, and out-of-fold predictions were generated.  Averaged test predictions from Level 0 models were used as input for the Level 1 Ridge Regression model.
3.  **Evaluation:** The final RMSE on the out-of-fold predictions was 0.108.

### Future Work
1.  **Hyperparameter Optimization:** Use Optuna or Bayesian optimization to tune base model hyperparameters.
2.  **Seed Averaging:** Retrain Level 0 models with different random seeds and average the final test predictions to improve robustness.
3.  **Leakage Mitigation:** Refactor the code such that imputation and scaling happens inside the CV folds.
```