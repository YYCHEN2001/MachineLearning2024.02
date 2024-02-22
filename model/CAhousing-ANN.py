import numpy as np  # 数据处理工具
import pandas as pd
from sklearn.metrics import mean_squared_error  # 引入均方差
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor  # MLP回归算法

np.random.seed(0)  # 保证每次数据唯一性

# 从本地获取california_housing.csv数据
df = pd.read_csv('./california_housing.csv')
X = df.iloc[:, :8]
y = df['target']

# 拆分数据集（train set & test set）
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

# # 导入神经网络
# from sklearn.neural_network import MLPRegressor 

# # 寻找合适的层数和元素值
# from sklearn.model_selection import GridSearchCV
# mlp = MLPRegressor(max_iter = 1000)

# # 生成隐藏层大小的列表
# # 单隐藏层：(1,) 到 (16,)
# # 双隐藏层：(1,1) 到 (16,16)
# # 三隐藏层：(1,1,1) 到 (16,16,16)
# hidden_layer_sizes = []
# for i in range(1, 17):  # 对于单隐藏层
#     hidden_layer_sizes.append((i,))
# for i in range(1, 17):  # 对于双隐藏层
#     for j in range(1, 17):
#         hidden_layer_sizes.append((i, j))
# for i in range(6, 17):  # 对于三隐藏层
#     for j in range(6, 17):
#        for k in range(6, 17):
#             hidden_layer_sizes.append((i, j, k))

# # 定义参数网格
# param_grid = {
#     'hidden_layer_sizes': hidden_layer_sizes
# }

# # 创建GridSearchCV对象
# grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1, cv=3, verbose=2)

# # 假设Xtrain和Ytrain是你的训练数据
# grid_search.fit(Xtrain, Ytrain)

# # 打印最佳参数
# print("最佳参数: ", grid_search.best_params_)

# 导入神经网络
reg = MLPRegressor(solver='adam',
                   alpha=1e-5,
                   hidden_layer_sizes=(16, 10, 6),
                   random_state=1)

# slover 是权重优化策略； 
# activation 表示选择的激活函数，这里没有设置，默认是 relu；
# alpha 是惩罚参数；
# hidden_layer_sizes 是隐藏层大小，长度就是隐藏层的数量，每一个大小就是设置每层隐藏层的神经元数量；
# random_state 是初始化所使用的随机项；

# 拟合
reg.fit(Xtrain, Ytrain)
y_pred = reg.predict(Xtrain)
y_test_pred = reg.predict(Xtest)

score = reg.score(Xtest, Ytest, sample_weight=None)
print('The predict value is ', y_test_pred)
print('The real value is ', Ytest)
print('Accuracy:', score)
print('layers nums :', reg.n_layers_)

# 模型评估，引入均方差MSE


# 输出训练集和测试集的均方误差
print("The MSE of train set is", mean_squared_error(Ytrain, y_pred))
print("The MSE of test set is", mean_squared_error(Ytest, y_test_pred))
