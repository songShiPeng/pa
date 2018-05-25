import xgboost as xgb
xr = xgb.XGBRegressor()
xr.fit(X, y)
xr.predict(X_test)