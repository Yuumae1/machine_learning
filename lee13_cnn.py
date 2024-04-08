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
data1 = np.load('/home/maeda/data/bsiso_lee13/lee13_rcnst.npz')
data2 = np.load('/home/maeda/data/bsiso_lee13/lee13_mveof.npz')
data3 = np.load('/home/maeda/data/bsiso_lee13/lee13_mveof_allperiod.npz')
data4 = np.load('/home/maeda/data/bsiso_lee13/pr_wtr.npz')
data_time = np.load('/home/maeda/data/bsiso_lee13/preprocessed_data_danomaly.npz')
print(data1.files)
print(data2.files)
print(data4.files)

lat = data2['lat'][20:42]
lon = data2['lon'][16:66]
#olr = data1['olr'][:,20:42,16:66]
#u850 = data1['u850'][:,20:42,16:66]
#v850 = data1['v850'][:,20:42,16:66]
#h850 = data1['h850'][:,20:42,16:66]
olr1 = data1['olr1']
olr2 = data1['olr2']
u8501 = data1['u8501']
u8502 = data1['u8502']
#v850 = data1['v850'][:,20:50,:]
h8501 = data1['h8501']
h8502 = data1['h8502']
#pr_wtr = data4['data_anom_rm'][:-365,20:50,16:66]
time = data_time['time']
real_time = pd.to_datetime(time, unit='h', origin=pd.Timestamp('1800-01-01')) # 時刻をdatetime型に変換
print(lat.shape, lon.shape, olr1.shape, time.shape, real_time.shape, u8501.shape, h8501.shape)
print(real_time[0], real_time[-1])

# bsiso index (MVEOF) 読み込み
PC1 = data2['PCs'][:,0]
PC2 = data2['PCs'][:,1]
PCs = np.stack([PC1, -PC2], axis=1)
#phase = data2['phase']
time3 = data3['time']

PCs = PCs / np.std(PCs, axis=0)
print(PCs.shape, time3.shape)
print(time3[0], time3[-1])

# 標準化処理
def normalization(data):
  data_std  = np.std(data, axis=0)
  data_norm = data / data_std
  print('Raw Data        = ', data.max(), data.min())
  print('Normalized Data = ', data_norm.max(), data_norm.min())
  del data_std
  return data_norm

olr1_norm = normalization(olr1)
olr2_norm = normalization(olr2)
u8501_norm = normalization(u8501)
u8502_norm = normalization(u8502)
#v850_norm = normalization(v850)
h8501_norm = normalization(h8501)
h8502_norm = normalization(h8502)
#pr_wtr_norm = normalization(pr_wtr)
del olr1, olr2, u8501, u8502, h8501, h8502

# initial の気象場を後ろにずらして予測問題を解くため、気象場の方をずらした後にインデクシングする
lead_time = 10
print(PCs.shape)
output_shape = 2
print('output shape = ', output_shape)

#ph = phase
rt = real_time
sup_data = PCs
print(sup_data.shape)

# 入力データの前処理
def preprocess(data):
  # np.roll は前方に要素をシフトさせるが、末端部分は巡回していることに注意する
  ipt_lag0   = np.roll(data, -lead_time, axis=0)
  #ipt_lag5   = np.roll(data, lead_time+5, axis=0)
  ipt_lag10  = np.roll(data, -lead_time-10, axis=0)
  # =========
  # 訓練データの作成(mjjaso)
  idx = np.where((rt.month >= 5) & (rt.month <= 10) & (rt.year <= 2015))[0]
  ipt_train_lag0 = ipt_lag0[idx]
  #ipt_train_lag5 = ipt_lag5[idx]
  ipt_train_lag10 = ipt_lag10[idx]
  ipt_train = np.stack([ipt_train_lag0, ipt_train_lag10],axis=3)

  # 検証データの作成
  idx = np.where((rt.month >= 5) & (rt.month <= 10) & (rt.year > 2015))[0]
  ipt_test_lag0 = ipt_lag0[idx]
  #ipt_test_lag5 = ipt_lag5[idx]
  ipt_test_lag10 = ipt_lag10[idx]
  ipt_test = np.stack([ipt_test_lag0, ipt_test_lag10], axis=3)
  return ipt_train, ipt_test

olr1_ipt_train, olr1_ipt_test = preprocess(olr1_norm)
olr2_ipt_train, olr2_ipt_test = preprocess(olr2_norm)
u8501_ipt_train, u8501_ipt_test = preprocess(u8501_norm)
u8502_ipt_train, u8502_ipt_test = preprocess(u8502_norm)
#v850_ipt_train, v850_ipt_test = preprocess(v850_norm)
h8501_ipt_train, h8501_ipt_test = preprocess(h8501_norm)
h8502_ipt_train, h8502_ipt_test = preprocess(h8502_norm)
#pr_wtr_ipt_train, pr_wtr_ipt_test = preprocess(pr_wtr_norm)

ipt_train = np.concatenate([olr1_ipt_train, olr2_ipt_train, u8501_ipt_train, u8502_ipt_train, 
                            h8501_ipt_train, h8502_ipt_train], 3)
ipt_test  = np.concatenate([olr1_ipt_test, olr2_ipt_test, u8501_ipt_test, u8502_ipt_test,
                            h8501_ipt_test, h8501_ipt_test], 3)
#ipt_train, ipt_test = v850_ipt_train, v850_ipt_test

# その他のインデクシング
sup_train = sup_data[:6808,:2]
sup_test = sup_data[6808:, :2]
idx = np.where((rt.month >= 5) & (rt.month <= 10) & (rt.year > 2015))[0]
rt = rt[idx]
print(sup_test.shape, sup_train.shape, ipt_test.shape, ipt_train.shape, rt.shape)
del olr1_ipt_train, olr2_ipt_train, u8501_ipt_train, u8502_ipt_train, h8501_ipt_train, h8502_ipt_train
# (1268, 2) (6808, 2) (1288, 22, 50, 12) (6808, 22, 50, 12)

# CNNモデルの構築 #0.4
model = Sequential()
# 入力画像　25×144×3 ：(緯度方向の格子点数)×(軽度方向の格子点数)×(チャンネル数、OLRのラグ)
model.add(Conv2D(32, (2, 2), padding='same', input_shape=(21, 49, 12), strides=(2,2) ))   # ゼロパディング、バッチサイズ以外の画像の形状を指定 25*144*1 -> 25*144*8
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
#model.add(Dense(1024))
model.add(Dense(128))
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