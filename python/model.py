# -*- coding: UTF-8 -*-
import pickle
import gzip

# 載入模型
with gzip.open('D:/python/xgboost-iris.pgz', 'rb') as f:
    xgboostModel = pickle.load(f)

# 將模型預測寫成一個 function 
def predict(input):
    pred=xgboostModel.predict(input)[0]
    return pred