import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
#读取数据
data=pd.read_csv('data.csv')
#观察过数据后，将特征字段分为3组
features_mean=list(data.columns[2:12])
features_se=list(data.columns[12:22])
features_worst=list(data.columns[22:32])
#数据清洗
#删除与转换
data.drop('id',axis=1,inplace=True)
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
#用热力图观察字段相关性
corr=data[features_mean].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,annot=True)
plt.show()
#观察数据并对数据降维处理，将原有10个维度降为6个
# 特征选择
features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']
# 抽取 30% 的数据作为测试集，其余作为训练集
train, test = train_test_split(data, test_size = 0.3)
# 抽取特征选择的数值作为训练和测试数据
train_X = train[features_remain]
train_y=train['diagnosis']
test_X= test[features_remain]
test_y =test['diagnosis']
# 采用 Z-Score 规范化数据，保证每个特征维度的数据均值为 0，方差为 1
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)
# 创建 SVM 分类器
model = svm.SVC()
# 用训练集做训练
model.fit(train_X,train_y)
# 用测试集做预测
prediction=model.predict(test_X)
print('准确率: ', metrics.accuracy_score(prediction,test_y))

