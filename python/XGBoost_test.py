import pickle
import gzip
import numpy as np

with gzip.open('D:/python/xgboost-iris.pgz', 'rb') as f:
    xgboostModel = pickle.load(f)
    pred=xgboostModel.predict(np.array([[5.5, 2.4, 3.7, 1. ]]))
    print(pred)