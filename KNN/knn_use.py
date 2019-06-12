#数据读取
import pandas as pd

features = ['accommodates','bedrooms','bathrooms','beds','price','minimum_nights','maximum_nights','number_of_reviews']

#读取csv文件
dc_listings = pd.read_csv('listings.csv')

dc_listings = dc_listings[features]
print(dc_listings.shape)

print(dc_listings.head())

#按照距离排序
import numpy as np

our_acc_value = 3   #房间数量与3的距离

dc_listings['distance'] = np.abs(dc_listings.accommodates - our_acc_value)    #比较距离
print(dc_listings.distance.value_counts().sort_index())    #排序

'''
#对数据进行洗牌操作
dc_listings = dc_listings.sample(frac=1,random_state=0)    #洗牌
dc_listings = dc_listings.sort_values('distance')
print(dc_listings.price.head())

#把price从字符串格式转换为float
dc_listings['price'] = dc_listings.price.str.replace("\$|,",'').astype(float)

mean_price = dc_listings.price.iloc[:5].mean()
print(mean_price)

#首先制定好训练集和测试集
dc_listings.drop('distance',axis=1)    #把distance那一列去掉

train_df = dc_listings.copy().iloc[:2792]
test_df = dc_listings.copy().iloc[2792:]

#基于单变量预测价格
def predict_price(new_listing_value,feature_column):
    temp_df = train_df
    temp_df['distance'] = np.abs(dc_listings[feature_column] - new_listing_value)
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return(predicted_price)

test_df['predicted_price'] = test_df.accommodates.apply(predict_price,feature_column='accommodates') #apply：对dataframe中的每一个样本都执行同样的操作
#print(test_df['predicted_price'])

#用RMSE衡量模型的好坏
test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**(2)
mse = test_df['squared_error'].mean()
rmse = mse ** (1/2)
print('rmse是：',rmse)

#不同的变量效果会不会不同呢？
for feature in ['accommodates','bedrooms','bathrooms','number_of_reviews']:
    test_df['predicted_price'] = test_df[feature].apply(predict_price,feature_column=feature)

    test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**(2)
    mse = test_df['squared_error'].mean()
    rmse = mse ** (1/2)
    print("RMSE for the {} column: {}".format(feature,rmse))
'''
#看起来结果差异还是蛮大的，接下来我们要做的就是综合利用所有的信息来一起进行测试
import pandas as pd
#sklearn经典的机器学习库
from sklearn.preprocessing import StandardScaler
features = ['accommodates','bedrooms','bathrooms','beds','price','minimum_nights','maximum_nights','number_of_reviews']

dc_listings = pd.read_csv('listings.csv')

dc_listings = dc_listings[features]

dc_listings['price'] = dc_listings.price.str.replace("\$|,",'').astype(float)

dc_listings = dc_listings.dropna()   #dropna把缺失值去掉

#StandardScaler标准化
dc_listings[features] = StandardScaler().fit_transform(dc_listings[features])

normalized_listings = dc_listings

print('dc_listings.shape是：',dc_listings.shape)

print('normalized_listings.head()是：',normalized_listings.head())


norm_train_df = normalized_listings.copy().iloc[0:2792]
norm_test_df = normalized_listings.copy().iloc[2792:]
'''
#scipy中已经有现成的距离的计算工具了
from scipy.spatial import distance    #计算欧式距离

first_listing = normalized_listings.iloc[0][['accommodates', 'bathrooms']]
fifth_listing = normalized_listings.iloc[20][['accommodates', 'bathrooms']]
first_fifth_distance = distance.euclidean(first_listing, fifth_listing)
print('first_fifth_distance是：',first_fifth_distance)

#多变量KNN模型
def predict_price_multivariate(new_listing_value,feature_columns):
    temp_df = norm_train_df
    temp_df['distance'] = distance.cdist(temp_df[feature_columns],[new_listing_value[feature_columns]])
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return(predicted_price)

cols = ['accommodates', 'bathrooms']
norm_test_df['predicted_price'] = norm_test_df[cols].apply(predict_price_multivariate,feature_columns=cols,axis=1)
norm_test_df['squared_error'] = (norm_test_df['predicted_price'] - norm_test_df['price'])**(2)
mse = norm_test_df['squared_error'].mean()
rmse = mse ** (1/2)
print(rmse)'''



#重点
#使用Sklearn来完成KNN
from sklearn.neighbors import KNeighborsRegressor   #用knn回归的模块
cols = ['accommodates','bedrooms']
knn = KNeighborsRegressor()   #可以传入k值，默认为5
knn.fit(norm_train_df[cols], norm_train_df['price'])   #fit()训练knn
two_features_predictions = knn.predict(norm_test_df[cols])   #测试

from sklearn.metrics import mean_squared_error

two_features_mse = mean_squared_error(norm_test_df['price'], two_features_predictions)
two_features_rmse = two_features_mse ** (1/2)
print('two_features_rmse是：',two_features_rmse)

#加入更多的特征
knn = KNeighborsRegressor()

cols = ['accommodates','bedrooms','bathrooms','beds','minimum_nights','maximum_nights','number_of_reviews']

knn.fit(norm_train_df[cols], norm_train_df['price'])
four_features_predictions = knn.predict(norm_test_df[cols])
four_features_mse = mean_squared_error(norm_test_df['price'], four_features_predictions)
four_features_rmse = four_features_mse ** (1/2)
print('four_features_rmse是：',four_features_rmse)