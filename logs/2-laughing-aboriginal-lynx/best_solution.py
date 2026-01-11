import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso
from scipy.stats import mstats
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings("ignore")

# Load data
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
sample_submission = pd.read_csv("input/sample_submission.csv")

train_id = train["Id"]
test_id = test["Id"]

train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

# Separate target variable
y = train["SalePrice"]
train.drop("SalePrice", axis=1, inplace=True)

# Combine train and test data
all_data = pd.concat([train, test], axis=0).reset_index(drop=True)

# Missing Value Handling
numerical_cols = all_data.select_dtypes(include=np.number).columns
categorical_cols = all_data.select_dtypes(exclude=np.number).columns

num_median_imputer = SimpleImputer(strategy="median")
num_constant_imputer = SimpleImputer(strategy="constant", fill_value=0)
num_mean_imputer = SimpleImputer(strategy="mean")
cat_most_frequent_imputer = SimpleImputer(strategy="most_frequent")

all_data["LotFrontage"] = num_median_imputer.fit_transform(all_data[["LotFrontage"]])[
    :, 0
]
all_data[["GarageYrBlt"]] = num_median_imputer.fit_transform(all_data[["GarageYrBlt"]])

cols_to_fill_with_zero = [
    "GarageCars",
    "GarageArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
]
all_data[cols_to_fill_with_zero] = num_constant_imputer.fit_transform(
    all_data[cols_to_fill_with_zero]
)

all_data["MSZoning"] = cat_most_frequent_imputer.fit_transform(all_data[["MSZoning"]])[
    :, 0
]
all_data["Utilities"] = cat_most_frequent_imputer.fit_transform(
    all_data[["Utilities"]]
)[:, 0]
all_data["Exterior1st"] = cat_most_frequent_imputer.fit_transform(
    all_data[["Exterior1st"]]
)[:, 0]
all_data["Exterior2nd"] = cat_most_frequent_imputer.fit_transform(
    all_data[["Exterior2nd"]]
)[:, 0]
all_data["MasVnrType"] = cat_most_frequent_imputer.fit_transform(
    all_data[["MasVnrType"]]
)[:, 0]
all_data["Electrical"] = cat_most_frequent_imputer.fit_transform(
    all_data[["Electrical"]]
)[:, 0]
all_data["KitchenQual"] = cat_most_frequent_imputer.fit_transform(
    all_data[["KitchenQual"]]
)[:, 0]
all_data["Functional"] = cat_most_frequent_imputer.fit_transform(
    all_data[["Functional"]]
)[:, 0]
all_data["GarageType"] = cat_most_frequent_imputer.fit_transform(
    all_data[["GarageType"]]
)[:, 0]
all_data["GarageFinish"] = cat_most_frequent_imputer.fit_transform(
    all_data[["GarageFinish"]]
)[:, 0]
all_data["GarageQual"] = cat_most_frequent_imputer.fit_transform(
    all_data[["GarageQual"]]
)[:, 0]
all_data["GarageCond"] = cat_most_frequent_imputer.fit_transform(
    all_data[["GarageCond"]]
)[:, 0]
all_data["BsmtQual"] = cat_most_frequent_imputer.fit_transform(all_data[["BsmtQual"]])[
    :, 0
]
all_data["BsmtCond"] = cat_most_frequent_imputer.fit_transform(all_data[["BsmtCond"]])[
    :, 0
]
all_data["BsmtExposure"] = cat_most_frequent_imputer.fit_transform(
    all_data[["BsmtExposure"]]
)[:, 0]
all_data["BsmtFinType1"] = cat_most_frequent_imputer.fit_transform(
    all_data[["BsmtFinType1"]]
)[:, 0]
all_data["BsmtFinType2"] = cat_most_frequent_imputer.fit_transform(
    all_data[["BsmtFinType2"]]
)[:, 0]
all_data["SaleType"] = cat_most_frequent_imputer.fit_transform(all_data[["SaleType"]])[
    :, 0
]

all_data.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1, inplace=True)

# Outlier Handling
y = np.log1p(y)

# Feature Transformation
ordinal_mapping = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ordinal_features = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "GarageQual",
    "GarageCond",
    "FireplaceQu",
]
for feature in ordinal_features:
    all_data[feature] = all_data[feature].map(ordinal_mapping).fillna(0)

# Feature Engineering
all_data["TotalSF"] = (
    all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
)
all_data["TotalBathrooms"] = (
    all_data["FullBath"]
    + 0.5 * all_data["HalfBath"]
    + all_data["BsmtFullBath"]
    + 0.5 * all_data["BsmtHalfBath"]
)
all_data["TotalPorchSF"] = (
    all_data["OpenPorchSF"]
    + all_data["EnclosedPorch"]
    + all_data["3SsnPorch"]
    + all_data["ScreenPorch"]
)
all_data["Age"] = 2024 - all_data["YearBuilt"]
all_data["RemodelAge"] = 2024 - all_data["YearRemodAdd"]
all_data["Age_Garage"] = 2024 - all_data["GarageYrBlt"]

all_data["OverallQual_GrLivArea"] = all_data["OverallQual"] * all_data["GrLivArea"]
all_data["TotalSF_OverallQual"] = all_data["TotalSF"] * all_data["OverallQual"]
all_data["LotArea_Neighborhood"] = (
    all_data["LotArea"] * all_data["Neighborhood"].astype("category").cat.codes
)
all_data["YearBuilt_OverallQual"] = all_data["YearBuilt"] * all_data["OverallQual"]

# Split back into train and test
X = all_data.iloc[: len(train), :]
X_test = all_data.iloc[len(train) :, :]

# One-Hot Encoding and Scaling
numerical_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(exclude=np.number).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", RobustScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough",
)

X = preprocessor.fit_transform(X)
X_test = preprocessor.transform(X_test)

# Feature Selection
selector = VarianceThreshold(threshold=0.01)
X = selector.fit_transform(X)
X_test = selector.transform(X_test)

# Model Training (XGBoost)
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.005,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Evaluate on validation set
predictions = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, predictions))
print(f"Validation RMSE: {rmse}")

# Make predictions on test data
test_predictions = model.predict(X_test)
test_predictions = np.expm1(test_predictions)

# Create submission file
submission = pd.DataFrame({"Id": test_id, "SalePrice": test_predictions})
submission.to_csv("working/submission.csv", index=False)
