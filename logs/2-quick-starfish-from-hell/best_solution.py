import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import skew
from scipy.special import boxcox1p

# 1. Data Loading and Initial Setup
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

train_ID = train["Id"]
test_ID = test["Id"]

train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

train = train[train["GrLivArea"] < 4000]

y = train["SalePrice"]
train.drop("SalePrice", axis=1, inplace=True)

all_data = pd.concat((train, test)).reset_index(drop=True)

# 2. Target Transformation
y_log = np.log1p(y)

# 3. Missing Value Imputation
# Categorical NaN meaning "None"
for col in (
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
):
    all_data[col] = all_data[col].fillna("None")

# Numerical NaN meaning 0
for col in (
    "GarageYrBlt",
    "GarageArea",
    "GarageCars",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
):
    all_data[col] = all_data[col].fillna(0)

# Mode Imputation
for col in (
    "MSZoning",
    "Electrical",
    "KitchenQual",
    "Exterior1st",
    "Exterior2nd",
    "SaleType",
):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# Special Cases
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data = all_data.drop(["Utilities"], axis=1)

# 4. Feature Engineering
all_data["TotalSF"] = (
    all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
)
all_data["TotalBath"] = (
    all_data["FullBath"]
    + 0.5 * all_data["HalfBath"]
    + all_data["BsmtFullBath"]
    + 0.5 * all_data["BsmtHalfBath"]
)
all_data["TotalPorchSF"] = (
    all_data["OpenPorchSF"]
    + all_data["3SsnPorch"]
    + all_data["EnclosedPorch"]
    + all_data["ScreenPorch"]
    + all_data["WoodDeckSF"]
)

all_data["HasPool"] = all_data["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
all_data["Has2ndfloor"] = all_data["2ndFlrSF"].apply(lambda x: 1 if x > 0 else 0)
all_data["HasGarage"] = all_data["GarageArea"].apply(lambda x: 1 if x > 0 else 0)
all_data["HasBsmt"] = all_data["TotalBsmtSF"].apply(lambda x: 1 if x > 0 else 0)
all_data["HasFireplace"] = all_data["Fireplaces"].apply(lambda x: 1 if x > 0 else 0)

all_data["MSSubClass"] = all_data["MSSubClass"].astype(str)
all_data["OverallCond"] = all_data["OverallCond"].astype(str)
all_data["YrSold"] = all_data["YrSold"].astype(str)
all_data["MoSold"] = all_data["MoSold"].astype(str)

# 5. Feature Transformation
# Ordinal Feature Mapping
quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
for col in (
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
    "PoolQC",
):
    all_data[col] = all_data[col].map(quality_map)

exposure_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
all_data["BsmtExposure"] = all_data["BsmtExposure"].map(exposure_map).fillna(0)

finish_map = {"GLQ": 5, "ALQ": 4, "BLQ": 3, "Rec": 2, "LwQ": 1, "Unf": 0, "None": 0}
for col in ["BsmtFinType1", "BsmtFinType2"]:
    all_data[col] = all_data[col].map(finish_map).fillna(0)

garage_finish_map = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}
all_data["GarageFinish"] = all_data["GarageFinish"].map(garage_finish_map).fillna(0)

# Skewed Feature Correction
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.5]
skewed_feats = skewed_feats.index

for feat in skewed_feats:
    all_data[feat] = boxcox1p(all_data[feat], 0.15)

# Categorical Feature Encoding
all_data = pd.get_dummies(all_data)

# 6. Data Splitting
X = all_data[: len(y_log)]
X_test = all_data[len(y_log) :]

# 7. Model Training
# Validation Framework
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Level 0: Base Models
lgb_model = lgb.LGBMRegressor(
    objective="regression_l1",
    n_estimators=2000,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=-1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    colsample_bytree=0.7,
    subsample=0.7,
    n_jobs=-1,
    random_state=42,
)

xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=4,
    colsample_bytree=0.7,
    subsample=0.7,
    reg_alpha=0.005,
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=100,
)

cat_model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.02,
    depth=6,
    l2_leaf_reg=3,
    loss_function="RMSE",
    early_stopping_rounds=100,
    verbose=False,
    random_state=42,
)

ridge_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=None)

# Level 0 Training and Prediction
lgb_oof = np.zeros(X.shape[0])
lgb_test_preds = np.zeros(X_test.shape[0])

xgb_oof = np.zeros(X.shape[0])
xgb_test_preds = np.zeros(X_test.shape[0])

cat_oof = np.zeros(X.shape[0])
cat_test_preds = np.zeros(X_test.shape[0])

ridge_oof = np.zeros(X.shape[0])
ridge_test_preds = np.zeros(X_test.shape[0])

for fold, (train_index, val_index) in enumerate(kf.split(X, y_log)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y_log.iloc[train_index], y_log.iloc[val_index]

    # LightGBM
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    lgb_oof[val_index] = lgb_model.predict(X_val)
    lgb_test_preds += lgb_model.predict(X_test) / kf.n_splits

    # XGBoost
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_oof[val_index] = xgb_model.predict(X_val)
    xgb_test_preds += xgb_model.predict(X_test) / kf.n_splits

    # CatBoost
    cat_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    cat_oof[val_index] = cat_model.predict(X_val)
    cat_test_preds += cat_model.predict(X_test) / kf.n_splits

    # Ridge
    ridge_model.fit(X_train, y_train)
    ridge_oof[val_index] = ridge_model.predict(X_val)
    ridge_test_preds += ridge_model.predict(X_test) / kf.n_splits

# Level 1: Meta-Model
X_meta = np.column_stack((lgb_oof, xgb_oof, cat_oof, ridge_oof))
X_test_meta = np.column_stack(
    (lgb_test_preds, xgb_test_preds, cat_test_preds, ridge_test_preds)
)

meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=None)
meta_model.fit(X_meta, y_log)
final_predictions = meta_model.predict(X_test_meta)

# 8. Evaluation
# Local Validation
oof_predictions = meta_model.predict(X_meta)
rmse = np.sqrt(mean_squared_error(y_log, oof_predictions))
print(f"RMSE on OOF Predictions: {rmse}")

# 9. Submission File Generation
final_predictions = np.expm1(final_predictions)
submission = pd.DataFrame({"Id": test_ID, "SalePrice": final_predictions})
submission.to_csv("working/submission.csv", index=False)
