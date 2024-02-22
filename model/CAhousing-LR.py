import time  # 计时模块
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 对训练集进行标准化

# 记录开始时间
start_time = time.time()
# 获取datasets.fetch_california_housing()数据
housing = datasets.fetch_california_housing()

# 特征解释
# MedInc：该街区住户的收入中位数
# HouseAge：该街区房屋使用年代的中位数
# AveRooms：该街区平均的房间数目
# AveBedrms：该街区平均的卧室数目
# Population：街区人口
# AveOccup：平均入住率
# Latitude：街区的纬度
# Longitude：街区的经度

# 将数据转化成DataFrame
X = pd.DataFrame(housing.data, columns=housing.feature_names)

# 获取目标值
y = housing.target

# 拆分数据集（train set & test set）
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

# 线性回归建模
lr = LinearRegression()

# 训练数据
lr.fit(Xtrain, Ytrain)

# 对训练集做预测
y_pred = lr.predict(Xtrain)
# 对测试集做预测
y_test_pred = lr.predict(Xtest)

# 输出训练集和测试集的均方误差
print("The MSE of train set is", mean_squared_error(Ytrain, y_pred))
print("The MSE of test set is", mean_squared_error(Ytest, y_test_pred))

# 输出交叉验证结果
# cv=10意为10折交叉验证
# scoring='neg_mean_squared_error'表示采用MSE作为指标
lr2 = LinearRegression()
print("The MSE of cross validation are", cross_val_score(lr2, Xtrain, Ytrain, cv=10, scoring='neg_mean_squared_error'))

# 输出交叉验证结果的平均值
print("The mean of the results above is",
      cross_val_score(lr2, Xtrain, Ytrain, cv=10, scoring='neg_mean_squared_error').mean())

# 输出训练集和测试集的R^2
print("\nThe R2 of train set is", r2_score(Ytrain, y_pred))
print("The R2 of test set is", r2_score(Ytest, y_test_pred))

# 同样可以使用交叉验证
print("The R2 of cross validation are", cross_val_score(lr, Xtrain, Ytrain, cv=10, scoring='r2'))
print("The mean of the results above is", cross_val_score(lr, Xtrain, Ytrain, cv=10, scoring='r2').mean())

# 查看模型系数
print("\nHere comes the results of model:")
print(list(zip(X.columns, lr.coef_)))

std = StandardScaler()
X_train_std = std.fit_transform(Xtrain[:])

# 将数据集标准化后再次训练
lr3 = LinearRegression()
lr3.fit(X_train_std, Ytrain)
# 输出标准化后的R2结果
print("\nAfter Normalization, the R2 of train set is", lr3.score(X_train_std, Ytrain))

# 设置图片dpi
plt.figure(figsize=(10, 9), dpi=300)
# 绘制散点图
plt.scatter(Ytest, y_test_pred, s=8)
# 绘制回归线
x_line = np.linspace(0, 5.1, 10)
y_line = x_line
plt.plot(x_line, y_line, 'r', linewidth=5.0, label="Regression Line")
# 设置x，y轴范围
plt.xlim(0, 5.1)
plt.ylim(-1, 7)
# 设置刻度字体大小
plt.tick_params(labelsize=20)
# x，y轴标签
plt.xlabel("True value", fontsize=24)
plt.ylabel("Predict value", fontsize=24)
# 显示图例
plt.legend()
# 显示图片
plt.show()

# 记录结束时间
end_time = time.time()
# 计算并输出运行时间
elapsed_time = end_time - start_time
print(f"Code execution time: {elapsed_time} seconds")
