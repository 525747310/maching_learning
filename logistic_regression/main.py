import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
path = 'data' + os.sep + 'LogiReg_data.txt'   #os.sep不同平台使用不同平台的分隔符
pdData = pd.read_csv(path, header=None, names=['Exams 1', 'Exams 2', 'Admitted'])   #读取csv文件,header自己指定names
print(pdData.head())
print(pdData.shape)

positive = pdData[pdData['Admitted'] == 1] #指定正例
negative = pdData[pdData['Admitted'] == 0]  #指定反例

fig, ax = plt.subplots(figsize=(10,5))   #figsize指定画图的大小
ax.scatter(positive[])

