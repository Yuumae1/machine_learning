import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import keras
import tensorflow
from numpy.linalg.linalg import norm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import activations
#from tf_keras_vis.activation_maximization import ActivationMaximization
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LayerNormalization
#from keras.optimizers import Adam
#from keras.optimizers import RMSprop
#from keras.regularizers import l2
#import matplotlib.pyplot as plt
#import matplotlib.ticker as mticker
#from matplotlib.markers import MarkerStyle
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
import pandas as pd
import shap
import numpy as np
#import cv2


# 画像用
from keras.preprocessing.image import array_to_img, img_to_array, load_img
# モデル読み込み用
from keras.models import load_model

# データの読み込み
data = np.load('/home/maeda/data/bsiso_eeof/prepro_anomaly_8vals.npz')
print('data = ', data.files)

lat = data['lat'][24:49]
lon = data['lon']
olr = data['olr'][80:,24:49,:]
u850 = data['u850'][80:,24:49,:]
v850 = data['v850'][80:,24:49,:]
u200 = data['u200'][80:,24:49,:]
v200 = data['v200'][80:,24:49,:]
h850 = data['h850'][80:,24:49,:]
pr_wtr = data['pr_wtr'][80:,24:49,:]
sst = data['sst'][80:,24:49,:]
time = data['time'][80:]    # 射影後にデータが10日進むため、時刻の方を前進させておく
real_time = pd.to_datetime(time, unit='h', origin=pd.Timestamp('1800-01-01')) # 時刻をdatetime型に変換
print(lat.shape, lon.shape, olr.shape, u850.shape, v850.shape, u200.shape, v200.shape, h850.shape, pr_wtr.shape)
print(real_time[0], real_time[-1])

# 標準化処理
def normalization(data):
  data_mean = np.mean(data, axis=0)
  data_std  = np.std(data, axis=0)
  data_norm = (data - data_mean) / data_std
  print('Raw Data        = ', data.max(), data.min())
  print('Normalized Data = ', data_norm.max(), data_norm.min())
  data_norm = np.nan_to_num(data_norm, nan=0) # 欠損値(nan)を0で置換
  del data_mean, data_std
  return data_norm

olr_norm  = normalization(olr)
u850_norm = normalization(u850)
v850_norm = normalization(v850)
u200_norm = normalization(u200)
v200_norm = normalization(v200)
h850_norm = normalization(h850)
pr_wtr_norm = normalization(pr_wtr)
sst_norm = normalization(sst)

# bsiso index (eEOF) 読み込み
data_file = '/home/maeda/data/bsiso_eeof/bsiso_rt-PCs.npz'
PC      = np.load(data_file)['rt_PCs'][:,:2]
sign    = np.array([-1, 1]).T
PC_norm = sign * PC / PC.std(axis=0)[np.newaxis,:]
time2   = np.load(data_file)['time']
real_time2 = pd.to_datetime(time2, unit='h', origin=pd.Timestamp('1800-01-01')) # 時刻をdatetime型に変換

print('PCs = ', PC_norm.shape)
print('time PCs= ', time2.shape)
print('real time PCs = ', real_time2[0], real_time2[-1])

# インデクシングする関数
def indexing(lead_time):
  output_shape = 2
  rt = real_time2[:-lead_time-1]
  sup_data = PC_norm[lead_time:]
  print(sup_data.shape)
  idx = np.where((rt.year <= 2015))[0]
  sup_train = sup_data[idx]
  idx = np.where((rt.year > 2015))[0]
  sup_test = sup_data[idx]
  print(sup_test.shape, sup_train.shape)
  return data, rt, sup_train, sup_test, output_shape

# 入力データの前処理
def preprocess(data, rt, lead_time):
  ipt_lag0  = data[10:-lead_time-1]
  ipt_lag5  = data[5:-lead_time-6]
  ipt_lag10 = data[:-lead_time-11]

  # 検証データの作成
  idx = np.where((rt.year > 2015))[0]
  ipt_lag0_test = ipt_lag0[idx]
  ipt_lag5_test = ipt_lag5[idx]
  ipt_lag10_test = ipt_lag10[idx]
  ipt_test = np.stack([ipt_lag0_test, ipt_lag5_test, ipt_lag10_test], 3)
  return ipt_test

# ==== iteration program ====
lt_box = [0]
#lt_box = [0, 5, 10, 15, 20, 25, 30, 35]
#lt_box = np.arange(36)
for lead_time in lt_box:

  print('==== lead time : {} day ====='.format(lead_time))

  data, rt, sup_train, sup_test, output_shape = indexing(lead_time) 

  olr_ipt_test = preprocess(olr_norm, rt, lead_time)
  u850_ipt_test = preprocess(u850_norm, rt, lead_time)
  v850_ipt_test = preprocess(v850_norm, rt, lead_time)
  u200_ipt_test = preprocess(u200_norm, rt, lead_time)
  v200_ipt_test = preprocess(v200_norm, rt, lead_time)
  h850_ipt_test = preprocess(h850_norm, rt, lead_time)
  pr_wtr_ipt_test = preprocess(pr_wtr_norm, rt, lead_time)
  sst_ipt_test = preprocess(sst_norm, rt, lead_time)

  ipt_test  = np.concatenate([olr_ipt_test, u850_ipt_test,  u200_ipt_test,
                              v850_ipt_test, v200_ipt_test, h850_ipt_test, 
                              pr_wtr_ipt_test, sst_ipt_test], 3)
  print(ipt_test.shape)


# モデルの読み込み
seed = 8
lead_time = 0
model_path = f'/home/maeda/machine_learning/results/model/kikuchi-8vals_v1/8vals/model_{(lead_time):03}day/seed{(seed):03}.hdf5'
model = load_model(model_path)

datasets = ipt_test
print(datasets.shape)
    
shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough # batch norm を挟む場合、このコードが必要：https://github.com/shap/shap/issues/1406
explainer = shap.DeepExplainer(model=model, data=datasets)
shap_values = explainer.shap_values(datasets, check_additivity=False)
shap_values = np.array(shap_values)
print('Deep Lift calculation is done!')
print(shap_values.shape)
print(shap_values.mean(axis=(0,1,2)))
np.savez(f'/home/maeda/machine_learning/results/kikuchi-8vals_v1/shap/shap_{lead_time}day.npz', shap_values=shap_values)