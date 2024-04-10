import numpy as np
from numpy.linalg.linalg import norm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LayerNormalization
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.regularizers import l2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.markers import MarkerStyle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd


# データの読み込み
data = np.load('/home/maeda/data/bsiso_lee13/prepro_anomaly_7vals.npz')
print('data = ', data.files)

lat = data['lat'][24:49]
lon = data['lon']
olr = data['olr'][80:-435,24:49,:]
u850 = data['u850'][80:-435,24:49,:]
v850 = data['v850'][80:-435,24:49,:]
u200 = data['u200'][80:-435,24:49,:]
v200 = data['v200'][80:-435,24:49,:]
h850 = data['h850'][80:-435,24:49,:]
pr_wtr = data['pr_wtr'][80:-435,24:49,:]
time = data['time'][80:-435]    # 射影後にデータが10日進むため、時刻の方を前進させておく
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
  del data_mean, data_std
  return data_norm

olr_norm  = normalization(olr)
u850_norm = normalization(u850)
v850_norm = normalization(v850)
u200_norm = normalization(u200)
v200_norm = normalization(v200)
h850_norm = normalization(h850)
pr_wtr_norm = normalization(pr_wtr)

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
  idx = np.where((rt.year <= 2014))[0]
  sup_train = sup_data[idx]
  idx = np.where((rt.year > 2014))[0]
  sup_test = sup_data[idx]
  print(sup_test.shape, sup_train.shape)
  return data, rt, sup_train, sup_test, output_shape

# 入力データの前処理
def preprocess(data, rt, lead_time):
  ipt_lag0  = data[10:-lead_time-1]
  ipt_lag5  = data[5:-lead_time-6]
  ipt_lag10 = data[:-lead_time-11]
  # =========
  # 訓練データの作成(通年データとする)
  idx = np.where((rt.year <= 2014))[0]
  ipt_lag0_train = ipt_lag0[idx]
  ipt_lag5_train = ipt_lag5[idx]
  ipt_lag10_train = ipt_lag10[idx]
  ipt_train = np.stack([ipt_lag0_train, ipt_lag5_train, ipt_lag10_train], 3)

  # 検証データの作成
  idx = np.where((rt.year > 2014))[0]
  ipt_lag0_test = ipt_lag0[idx]
  ipt_lag5_test = ipt_lag5[idx]
  ipt_lag10_test = ipt_lag10[idx]
  #ipt_test = ipt[idx]
  #ipt_test = np.concatenate([ipt_lag0_test, ipt_lag5_test, ipt_lag10_test], 1)
  ipt_test = np.stack([ipt_lag0_test, ipt_lag5_test, ipt_lag10_test], 3)
  return ipt_train, ipt_test





# CNNモデルの構築
def cnn_model():
  model = Sequential()
  # 入力画像　25×144×3 ：(緯度方向の格子点数)×(軽度方向の格子点数)×(チャンネル数、OLRのラグ)
  model.add(Conv2D(32, (3, 3), padding='same', input_shape=(25, 144, 3*7), strides=(2,2) ))   
  model.add(LayerNormalization())
  model.add(Activation('relu'))                                           
  model.add(Conv2D(64, (2, 2), padding='same', strides=(2,2)))                                        
  model.add(LayerNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(128, (2, 2), padding='same', strides=(2,2)))                           
  model.add(LayerNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.2)) 

  model.add(Flatten())  # 一次元の配列に変換                                # 1*16*64 -> 1024
  model.add(Dense(64))
  model.add(Activation('relu'))
  #model.add(Dense(64))
  model.add(Dense(output_shape=2, activation='linear'))
  model.summary()
  return model

def culc_cor(predict, y_test, lead_time):
  cor = (np.sum(predict[:,0] * y_test[:,0], axis=0) + np.sum(predict[:,1] * y_test[:,1], axis=0)) / \
          (np.sqrt(np.sum(predict[:,0] ** 2 + predict[:,1] ** 2, axis=0)) * np.sqrt(np.sum(y_test[:,0] ** 2 + y_test[:,1] ** 2, axis=0)))
  print('lead time {} day = '.format(lead_time), cor)

def learning_curve(history, lead_time):
  plt.figure(figsize=(8, 6))
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')    #Validation loss : 精度検証データにおける損失
  plt.xlim(0, 500)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Loss vs. Epoch   Lead Time = ' + str(lead_time) + 'days')
  plt.legend()
  plt.savefig('/home/maeda/machine_learning/results/kikuchi-7vals_v1/learning_curve_7vals_' + str(lead_time) + 'day.png')
  plt.close()


# ==== iteration program ====

lead_time = 0
print('==== lead time = {} day'.format(lead_time))

data, rt, sup_train, sup_test, output_shape = indexing(lead_time)

olr_ipt_train, olr_ipt_test = preprocess(olr_norm, rt, lead_time)
u850_ipt_train, u850_ipt_test = preprocess(u850_norm, rt, lead_time)
v850_ipt_train, v850_ipt_test = preprocess(v850_norm, rt, lead_time)
u200_ipt_train, u200_ipt_test = preprocess(u200_norm, rt, lead_time)
v200_ipt_train, v200_ipt_test = preprocess(v200_norm, rt, lead_time)
h850_ipt_train, h850_ipt_test = preprocess(h850_norm, rt, lead_time)
pr_wtr_ipt_train, pr_wtr_ipt_test = preprocess(pr_wtr_norm, rt, lead_time)

ipt_train = np.concatenate([olr_ipt_train, u850_ipt_train,
                            v850_ipt_train, u200_ipt_train, v200_ipt_train, h850_ipt_train, pr_wtr_ipt_train
                            ], 3)
ipt_test  = np.concatenate([olr_ipt_test, u850_ipt_test, 
                            v850_ipt_test, u200_ipt_test, v200_ipt_test, h850_ipt_test, pr_wtr_ipt_test
                            ], 3)
print(ipt_train.shape, ipt_test.shape)


model = cnn_model()
model.compile(optimizer=Adam(), loss='mean_squared_error')
history = model.fit(ipt_train, sup_train, epochs=10, batch_size=128, validation_data=(ipt_test, sup_test))
predict = model.predict(ipt_test, batch_size=None, verbose=0, steps=None) # モデルの出力を獲得する
print(predict.shape)
y_test = sup_test
culc_cor(predict, y_test, lead_time)
learning_curve(history, lead_time)
np.savez('/home/maeda/machine_learning/results/kikuchi-7vals_v1/result-value_7vals_' + str(lead_time) + 'day.npz', predict, y_test)
model.save('/home/maeda/machine_learning/results/kikuchi-7vals_v1/model_7vals_' + str(lead_time) + 'day.hdf5')
