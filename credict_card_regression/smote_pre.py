#SMOTE过采样方案
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#读数据
credit_cards=pd.read_csv('creditcard.csv')

columns=credit_cards.columns

# 在特征中去除掉标签
features_columns=columns.delete(len(columns)-1)

features=credit_cards[features_columns]
labels=credit_cards['Class']

features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                            labels,
                                                                            test_size=0.3,
                                                                            random_state=0)

#基于SMOTE算法来进行样本生成，这样正例和负例样本数量就是一致的了
oversampler=SMOTE(random_state=0)
os_features,os_labels=oversampler.fit_sample(features_train,labels_train)

#训练集样本数量
len(os_labels[os_labels==1])



