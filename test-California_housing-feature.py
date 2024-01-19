import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

#获取数据
from sklearn import datasets
housing = datasets.fetch_california_housing()
df = pd.DataFrame(housing.data,columns=housing.feature_names)
df['y'] = housing.target

# 计算出相关系数并输出，这里选择的是皮尔逊相关系数
cor = df.corr(method='pearson')
print(cor)  # 输出相关系数

