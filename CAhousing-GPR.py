# 计时模块
import time
# 记录开始时间
start_time = time.time() 

from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score



# 1. 加载加利福尼亚房屋数据集
housing = datasets.fetch_california_housing()

# 2. 数据集分割
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size = 0.3, random_state = 42)

# 3. 使用高斯过程回归模型
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel = kernel, random_state = 0)

# 4. 训练模型
gpr.fit(X_train, y_train)

# 5. 进行预测
y_pred = gpr.predict(X_test)

# 6. 使用R2和MSE评估模型性能
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R^2: {r2}, MSE: {mse}")

# 7. 进行5折交叉验证
scores = cross_val_score(gpr, housing.data, housing.target, cv=5, scoring='neg_mean_squared_error')
print(f"5-fold Cross Validation MSE: {scores}")

# 记录结束时间
end_time = time.time()

# 计算并输出运行时间
elapsed_time = end_time - start_time
print(f"Code execution time: {elapsed_time} seconds")