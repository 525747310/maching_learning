#%matplotlib inline

import matplotlib.pyplot as plt

import pandas as pd

#sklearn.datasets内置数据集
from sklearn.datasets.california_housing import fetch_california_housing   #房价数据集
housing = fetch_california_housing()
print('housing.DESCR:',housing.DESCR)

print('housing.data.shape:',housing.data.shape)

print('housing.data[0]',housing.data[0])

#使用决策树模块
from sklearn import tree
dtr = tree.DecisionTreeRegressor(max_depth = 2)   #树的最大深度
dtr.fit(housing.data[:, [6, 7]], housing.target)   #.fit(x,y)

#要可视化显示 首先需要安装 graphviz   http://www.graphviz.org
#https://blog.csdn.net/hawk_2016/article/details/82254228
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38/bin'  #注意修改你的路径
dot_data = \
    tree.export_graphviz(
        dtr,   #决策树对象
        out_file = None,
        feature_names = housing.feature_names[6:8],    #特征名字
        filled = True,
        impurity = False,
        rounded = True
    )

#pip install pydotplus
import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor("#FFF2DD")   #指定颜色
from IPython.display import Image
Image(graph.create_png())
graph.write_png("dtr_white_background.png")   #保存到本地

#切分数据集
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = \
    train_test_split(housing.data, housing.target, test_size = 0.1, random_state = 42)    #取10%作为测试集
#为了便于复现代码
dtr = tree.DecisionTreeRegressor(random_state = 42)    #random_state每次执行代码，随机完后的结果相同
dtr.fit(data_train, target_train)

dtr.score(data_test, target_test)


#随机森林
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor( random_state = 42)
rfr.fit(data_train, target_train)
rfr.score(data_test, target_test)

#通过for循环尝试多种参数的组合来查看怎样的参数选择效果最好
from sklearn.model_selection import GridSearchCV
tree_param_grid = { 'min_samples_split': list((3,6,9)),'n_estimators':list((10,50,100))}
grid = GridSearchCV(RandomForestRegressor(),param_grid=tree_param_grid, cv=5)    #cv=5进行多少次的交叉验证
grid.fit(data_train, target_train)
print(grid.cv_results_, grid.best_params_, grid.best_score_)

'''
rfr = RandomForestRegressor( min_samples_split=3,n_estimators = 100,random_state = 42)
rfr.fit(data_train, target_train)
rfr.score(data_test, target_test)

pd.Series(rfr.feature_importances_, index = housing.feature_names).sort_values(ascending = False)'''