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
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.markers import MarkerStyle
import pandas as pd


# データの読み込み
data1 = np.load('/home/maeda/data/bsiso_lee13/preprocessed_data_danomaly.npz')
data2 = np.load('/home/maeda/data/bsiso_lee13/lee13_mveof.npz')
data3 = np.load('/home/maeda/data/bsiso_lee13/lee13_mveof_allperiod.npz')
data4 = np.load('/home/maeda/data/bsiso_lee13/pr_wtr.npz')
print(data1.files)
print(data2.files)
print(data4.files)

lat = data2['lat'][20:42]
lon = data2['lon'][16:66]
#olr = data1['olr'][:,20:42,16:66]
#u850 = data1['u850'][:,20:42,16:66]
#v850 = data1['v850'][:,20:42,16:66]
#h850 = data1['h850'][:,20:42,16:66]
olr = data1['olr']
u850 = data1['u850']
#v850 = data1['v850'][:,20:50,:]
h850 = data1['h850']
#pr_wtr = data4['data_anom_rm'][:-365,20:50,16:66]
time = data1['time']
real_time = pd.to_datetime(time, unit='h', origin=pd.Timestamp('1800-01-01')) # 時刻をdatetime型に変換
print(lat.shape, lon.shape, olr.shape, time.shape, real_time.shape, u850.shape, h850.shape)
print(real_time[0], real_time[-1])

# bsiso index (MVEOF) 読み込み
PC1 = data3['PCs'][:,0]
PC2 = data3['PCs'][:,1]
PCs = np.stack([PC1, -PC2], axis=1)
#phase = data2['phase']
time3 = data3['time']

PCs = PCs / np.std(PCs, axis=0)
print(PCs.shape, time3.shape)
print(time3[0], time3[-1])

# 標準化処理
def normalization(data):
  data_std  = np.std(data, axis=0)
  data_mean = np.mean(data, axis=0)
  data_norm = (data-data_mean) / data_std
  print('Raw Data        = ', data.max(), data.min())
  print('Normalized Data = ', data_norm.max(), data_norm.min())
  del data_std, data_mean
  return data_norm

olr_norm = normalization(olr)
u850_norm = normalization(u850)
h850_norm = normalization(h850)
#pr_wtr_norm = normalization(pr_wtr)
del olr, olr, u850, u850, h850

# initial の気象場を後ろにずらして予測問題を解くため、気象場の方をずらした後にインデクシングする
lead_time = 10
print(PCs.shape)
output_shape = 2
print('output shape = ', output_shape)

#ph = phase
rt = real_time
# 教師データを前進
sup_data = PCs[10+lead_time:]
print(sup_data.shape)

# 入力データの前処理

def preprocess(data):
  ipt_lag0  = data[:-lead_time-1]
  #ipt_lag5  = data[5:-lead_time-6]
  #ipt_lag10 = data[:-lead_time-11]

  # =========
  # 訓練データの作成
  idx = np.where((rt.month >= 5) & (rt.month <= 10) & (rt.year <= 2015))[0]
  ipt_lag0_train = ipt_lag0[idx]
  #ipt_lag5_train = ipt_lag5[idx]
  #ipt_lag10_train = ipt_lag10[idx]
  #ipt_train = np.concatenate([ipt_lag0_train, ipt_lag5_train, ipt_lag10_train], 1)
  #ipt_train = np.stack([ipt_lag0_train, ipt_lag5_train, ipt_lag10_train], 3)
  ipt_train = ipt_lag0_train

  # 検証データの作成
  idx = np.where((rt.month >= 5) & (rt.month <= 10) & (rt.year > 2015))[0]
  ipt_lag0_test = ipt_lag0[idx]
  #ipt_lag5_test = ipt_lag5[idx]
  #ipt_lag10_test = ipt_lag10[idx]
  #ipt_test = np.concatenate([ipt_lag0_test, ipt_lag5_test, ipt_lag10_test], 1)
  #ipt_test = np.stack([ipt_lag0_test, ipt_lag5_test, ipt_lag10_test], 3)
  ipt_test = ipt_lag0_test
  return ipt_train, ipt_test

olr1_ipt_train, olr1_ipt_test = preprocess(olr_norm)
u8501_ipt_train, u8501_ipt_test = preprocess(u850_norm)
#v850_ipt_train, v850_ipt_test = preprocess(v850_norm)
h8501_ipt_train, h8501_ipt_test = preprocess(h850_norm)
#pr_wtr_ipt_train, pr_wtr_ipt_test = preprocess(pr_wtr_norm)

ipt_train = np.stack([olr1_ipt_train, u8501_ipt_train, 
                            h8501_ipt_train], 3)
ipt_test  = np.stack([olr1_ipt_test, u8501_ipt_test,
                            h8501_ipt_test], 3)
#ipt_train, ipt_test = v850_ipt_train, v850_ipt_test

# その他のインデクシング
idx = np.where((rt.month >= 5) & (rt.month <= 10) & (rt.year <= 2015))[0]
sup_train = sup_data[idx]
idx = np.where((rt.month >= 5) & (rt.month <= 10) & (rt.year > 2015))[0]
sup_test = sup_data[idx]
rt = rt[idx]
print(sup_test.shape, sup_train.shape, ipt_test.shape, ipt_train.shape, rt.shape)
del olr1_ipt_train, u8501_ipt_train, h8501_ipt_train
# (1268, 2) (6808, 2) (1288, 22, 50, 12) (6808, 22, 50, 12)

# CNNモデルの構築 #0.4
model = Sequential()
# 入力画像　25×144×3 ：(緯度方向の格子点数)×(軽度方向の格子点数)×(チャンネル数、OLRのラグ)
model.add(Conv2D(32, (2, 2), padding='same', input_shape=(21, 49, 6), strides=(2,2) ))   # ゼロパディング、バッチサイズ以外の画像の形状を指定 25*144*1 -> 25*144*8
model.add(LayerNormalization())
model.add(Activation('relu'))                                             # 活性化関数
#model.add(MaxPooling2D(pool_size=(2, 2)))                                 # 21*140*16 -> 10*70*16
model.add(Conv2D(64, (2, 2), padding='same', strides=(2,2)))                                             # 25*144*8 -> 21*140*16
model.add(LayerNormalization())
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))                                 # 21*140*16 -> 10*70*16

model.add(Conv2D(64, (2, 2), padding='same', strides=(2,2)))                             # 10*70*16 -> 10*70*32
model.add(LayerNormalization())
model.add(Activation('relu'))

model.add(Flatten())  # 一次元の配列に変換                                # 1*16*64 -> 1024
model.add(Dense(1024))
model.add(Dense(256))
#model.add(Activation('relu'))
#model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(output_shape, activation='linear'))
model.summary()

# モデルのコンパイルと学習の設定
# !!! acuracy = 0.20を目指す !!
# lead_time = 0 day で val_loss < 0.1 が望ましい
callback = keras.callbacks.EarlyStopping(monitor='loss',patience=3)
model.compile(optimizer=Adam(),loss='mean_squared_error')
# モデルのトレーニング
history = model.fit(ipt_train, sup_train, epochs=100, batch_size=128, validation_data=(ipt_test, sup_test), callbacks=[callback])

# モデルの出力を獲得する
predict = model.predict(ipt_test, batch_size=None, verbose=0, steps=None)
print(predict.shape)
y_test = sup_test
np.savez('/home/maeda/machine_learning/results/cnn-2d/' + str(lead_time) + 'day.npz', predict, y_test)

# モデルの保存
model.save('/home/maeda/machine_learning/results/cnn-2d/' + str(lead_time) + 'day.hdf5')