import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge  # 引入核岭回归
KRR = KernelRidge()

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

# 将数据转化成DataFrame,去除最后三个feature
X = pd.DataFrame(housing.data, columns=housing.feature_names)

# 获取目标值
y = housing.target

# 拆分数据集（train set & test set）
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

# 训练模型
KRR.fit(Xtrain, Ytrain)

# 对训练集做预测
y_pred = KRR.predict(Xtrain)
# 对测试集做预测
y_test_pred = KRR.predict(Xtest)

# 输出训练集和测试集的均方误差
print("The MSE of train set is", mean_squared_error(Ytrain, y_pred))
print("The MSE of test set is", mean_squared_error(Ytest, y_test_pred))

# 输出交叉验证结果
# cv=10意为10折交叉验证
# scoring='neg_mean_squared_error'表示采用MSE作为指标
KRR2 = KernelRidge()
MSE_cv = cross_val_score(KRR2, Xtrain, Ytrain, cv=5, scoring='neg_mean_squared_error')
print("The MSE of cross validation are", MSE_cv)

# 输出交叉验证结果的平均值
print("The mean of the results above is", MSE_cv.mean())

# 输出训练集和测试集的R^2
print("\nThe R2 of train set is", r2_score(Ytrain, y_pred))
print("The R2 of test set is", r2_score(Ytest, y_test_pred))

# 同样可以使用交叉验证
R2_MSE = cross_val_score(KRR2, Xtrain, Ytrain, cv=5, scoring='r2')
print("The R2 of cross validation are", R2_MSE)
print("The mean of the results above is", R2_MSE.mean())

# 查看模型系数
print("\nHere comes the results of model:")
print(list(zip(X.columns, KRR.dual_coef_)))

# 数据标准化
std = StandardScaler()
X_train_std = std.fit_transform(Xtrain[:])

# 将数据集标准化后再次训练
KRR3 = KernelRidge()
KRR3.fit(X_train_std, Ytrain)
# 输出标准化后的R2结果
y_std_pred = KRR3.predict(X_train_std)
print("\nAfter Normalization, the R2 of train set is", r2_score(Ytrain, y_std_pred))

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
