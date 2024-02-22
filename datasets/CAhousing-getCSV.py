import pandas as pd
import time
from sklearn import datasets

# 记录开始时间
start_time = time.time()
# 获取加州房价数据集
housing = datasets.fetch_california_housing()
# 将数据转化为DataFrame，并将target列添加到最后
df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
df['target'] = housing.target
# 输出为csv文件
csv_file_path = "./california_housing.csv"
df.to_csv(csv_file_path, index=False)
# 记录结束时间
end_time = time.time()
# 计算并输出运行时间
elapsed_time = end_time - start_time
print(f"Code execution time: {elapsed_time} seconds")
