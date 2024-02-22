import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time

# 开始计时
start_time = time.time()

# 加载数据
df = pd.read_csv('./california_housing.csv')
X = df.iloc[:, :8]
y = df['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练随机森林回归模型
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# 预测和评估
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# 评价指标
r2_train = r2_score(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)

# 使用5折交叉验证评估模型性能
cv_r2_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
cv_mse_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 训练随机森林回归模型（标准化数据）
rf_scaled = RandomForestRegressor(random_state=42)
rf_scaled.fit(X_train_scaled, y_train)

# 预测和评估（标准化数据）
y_pred_train_scaled = rf_scaled.predict(X_train_scaled)
y_pred_test_scaled = rf_scaled.predict(X_test_scaled)

# 评价指标（标准化数据）
r2_train_scaled = r2_score(y_train, y_pred_train_scaled)
mse_train_scaled = mean_squared_error(y_train, y_pred_train_scaled)
r2_test_scaled = r2_score(y_test, y_pred_test_scaled)
mse_test_scaled = mean_squared_error(y_test, y_pred_test_scaled)

# 使用5折交叉验证评估模型性能（标准化数据）
cv_r2_scores_scaled = cross_val_score(rf_scaled, X_scaled, y, cv=5, scoring='r2')
cv_mse_scores_scaled = cross_val_score(rf_scaled, X_scaled, y, cv=5, scoring='neg_mean_squared_error')

# 结束计时
end_time = time.time()
elapsed_time = end_time - start_time

# 输出结果
results = pd.DataFrame({
    'Metric': ['R2 (Train)', 'MSE (Train)', 'R2 (Test)', 'MSE (Test)', 'CV R2', 'CV MSE'],
    'Original Data': [r2_train, mse_train, r2_test, mse_test, np.mean(cv_r2_scores), -np.mean(cv_mse_scores)],
    'Standardized Data': [r2_train_scaled, mse_train_scaled, r2_test_scaled,
                          mse_test_scaled, np.mean(cv_r2_scores_scaled), -np.mean(cv_mse_scores_scaled)]
})

print(results)
print(f"Elapsed time: {elapsed_time} seconds")
