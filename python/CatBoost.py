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

housing.isnull().sum()

sns.set(rc={'figure.figsize':(10,10)})
# 使用的資料是房價MEDIV
sns.histplot(housing['MEDV'], kde=True)

correlation_matrix = housing.corr().round(2)
# annot = True 讓我們可以把數字標進每個格子裡
sns.heatmap(data=correlation_matrix, annot = True)
plt.show()

from sklearn.model_selection import train_test_split
X  = housing.drop(['MEDV'],axis=1).values
y = housing['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# print('Training data shape:',X_train.shape)
# print('Testing data shape:',X_test.shape)

from catboost import CatBoostRegressor
# 建立模型
model = CatBoostRegressor(random_state=42,
                         loss_function='RMSE',
                         eval_metric='RMSE',
                         use_best_model=True)
# 使用訓練資料訓練模型
model.fit(X_train,y_train, eval_set=(X_test, y_test), verbose=0, plot=True)
# print("Score: ", model.score(X_test, y_test))

feature_names = np.array(housing_dataset.feature_names)

# 確保 feature_importances_ 和 sorted_feature_importance 正確生成
sorted_feature_importance = model.feature_importances_.argsort()

# 繪製水平條形圖
plt.barh(feature_names[sorted_feature_importance], 
         model.feature_importances_[sorted_feature_importance], 
         color='turquoise')
plt.xlabel("CatBoost Feature Importance")
plt.show()

from catboost import Pool
import shap
shap.initjs()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# summarize the effects of all the features
shap.summary_plot(shap_values, X_test, feature_names=housing_dataset.feature_names)

from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
# feature_names = ['F{}'.format(i) for i in range(X_train.shape[1])]
train_pool = Pool(X_train, y_train, feature_names=housing_dataset.feature_names)
test_pool = Pool(X_test, y_test, feature_names=housing_dataset.feature_names)

model = CatBoostRegressor(random_state=42,
                         loss_function='RMSE',
                         eval_metric='RMSE',
                         use_best_model=True)
summary = model.select_features(
    train_pool,
    eval_set=test_pool,
    features_for_select='0-7',
    num_features_to_select=3,
    steps=2,
    algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
    shap_calc_type=EShapCalcType.Regular,
    train_final_model=True,
    logging_level='Silent',
    plot=False
)
summary
print(summary)

from catboost import CatBoostRegressor
grid = {'iterations': [100, 150, 200],
        'learning_rate': [0.03, 0.1],
        'depth': [2, 4, 6, 8],
        'l2_leaf_reg': [0.2, 0.5, 1, 3]}

model = CatBoostRegressor(random_state=42,
                         loss_function='RMSE',
                         eval_metric='RMSE')
model.grid_search(grid, X_train,y_train)