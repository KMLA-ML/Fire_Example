from xgboost import XGBRegressor

xgbr = XGBRegressor(max_depth=1, learning_rate=0.01, n_estimators=500)
xgbr.fit(X_train, y_train)

print(mean_squared_error(xgbr.predict(X_test), y_test))