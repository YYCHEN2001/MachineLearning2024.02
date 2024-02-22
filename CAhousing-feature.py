import pandas as pd
from sklearn import datasets

# 获取数据
housing = datasets.fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['y'] = housing.target

# 计算出相关系数并输出，这里选择的是皮尔逊相关系数
cor = df.corr(method='pearson')
print(cor)  # 输出相关系数
