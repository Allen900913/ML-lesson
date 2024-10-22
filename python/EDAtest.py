import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


np.set_printoptions(suppress=True)

iris = load_iris()
df_data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])
df_data

# 檢查缺失值
X = df_data.drop(labels=['Species'],axis=1).values # 移除Species並取得剩下欄位資料
y = df_data['Species']

#  切割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardization平均&變異數標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print('train shape:', X_train.shape)
print('test shape:', X_test.shape)
# checked missing data
print("checked missing data(NAN mount):",len(np.where(np.isnan(X))[0]))

# scaled之後的資料零均值，單位方差  
print('資料集 X 的平均值 : ', X_train.mean(axis=0))
print('資料集 X 的標準差 : ', X_train.std(axis=0))

print('\nStandardScaler 縮放過後訓練集的平均值 : ', X_train_scaled.mean(axis=0))
print('StandardScaler 縮放過後訓練集的標準差 : ', X_train_scaled.std(axis=0))



# fig, axes = plt.subplots(nrows=1,ncols=4)
# fig.set_size_inches(15, 4)
# sns.histplot(df_data["SepalLengthCm"][:],ax=axes[0], kde=True)
# sns.histplot(df_data["SepalWidthCm"][:],ax=axes[1], kde=True)
# sns.histplot(df_data["PetalLengthCm"][:],ax=axes[2], kde=True)
# sns.histplot(df_data["PetalWidthCm"][:],ax=axes[3], kde=True)

# correlation 計算
# corr = df_data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']].corr()
# plt.figure(figsize=(8,8))
# sns.heatmap(corr, square=True, annot=True, cmap="RdBu_r")

# plt.show()
