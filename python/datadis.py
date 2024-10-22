import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
from sklearn.datasets import fetch_california_housing
import warnings
warnings.filterwarnings('ignore')

housing_dataset = fetch_california_housing()


housing = pd.DataFrame(housing_dataset.data, columns=housing_dataset.feature_names)
housing['MEDV'] = housing_dataset.target
housing.head()

housing.describe()

plt.figure(figsize=(2,5))
plt.boxplot(housing['MedInc'],showmeans=True)
plt.title('MedInc')
plt.show()

skewness = round(housing['MedInc'].skew(), 2)
kurtosis = round(housing['MedInc'].kurt(), 2)
print(f"偏度(Skewness): {skewness}, 峰度(Kurtosis): {kurtosis}")

# 繪製分布圖
sns.histplot(housing['MedInc'], kde=True)
plt.show()

transform_data = np.log(housing['MedInc'])
# skewness 與 kurtosis
skewness = round(transform_data.skew(), 2)
kurtosis = round(transform_data.kurt(), 2)
print(f"偏度(Skewness): {skewness}, 峰度(Kurtosis): {kurtosis}")

# 繪製分布圖
sns.histplot(transform_data, kde=True)
plt.show()
  
# 將所有特徵超出1.5倍IQR的概念將這些Outlier先去掉，避免對Model造成影響
print ("Shape Of The Before Ouliers: ",housing['MedInc'].shape)
n=1.5
#IQR = Q3-Q1
IQR = np.percentile(housing['MedInc'],75) - np.percentile(housing['MedInc'],25)
# outlier = Q3 + n*IQR 
transform_data=housing[housing['MedInc'] < np.percentile(housing['MedInc'],75)+n*IQR]
# outlier = Q1 - n*IQR 
transform_data=transform_data[transform_data['MedInc'] > np.percentile(transform_data['MedInc'],25)-n*IQR]['MedInc']
print ("Shape Of The After Ouliers: ",transform_data.shape)
plt.figure(figsize=(2,5))
plt.boxplot(transform_data,showmeans=True)
plt.title('MedInc')
plt.show()