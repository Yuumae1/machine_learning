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
import numpy as np


# 画像用
from keras.preprocessing.image import array_to_img, img_to_array, load_img
# モデル読み込み用
from keras.models import load_model
# Grad−CAM計算用
from tensorflow.keras import models
import tensorflow as tf

# データの読み込み
data = np.load('/home/maeda/data/bsiso_eeof/prepro_anomaly_7vals.npz')
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
  del data_mean, data_std
  return data_norm

olr_norm  = normalization(olr)
u850_norm = normalization(u850)
#v850_norm = normalization(v850)
u200_norm = normalization(u200)
#v200_norm = normalization(v200)
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
  idx = np.where((rt.year <= 2015))[0]
  sup_train = sup_data[idx]
  idx = np.where((rt.year > 2015))[0]
  sup_test = sup_data[idx]
  rt_test = rt[idx]
  print(sup_test.shape, sup_train.shape)
  return data, rt, sup_train, sup_test, output_shape, rt_test

# 入力データの前処理
def preprocess(data, rt, lead_time):
  ipt_lag0  = data[10:-lead_time-1]
  ipt_lag5  = data[5:-lead_time-6]
  ipt_lag10 = data[:-lead_time-11]
  # =========
  # 訓練データの作成(通年データとする)
  idx = np.where((rt.year <= 2015))[0]
  ipt_lag0_train = ipt_lag0[idx]
  ipt_lag5_train = ipt_lag5[idx]
  ipt_lag10_train = ipt_lag10[idx]
  ipt_train = np.stack([ipt_lag0_train, ipt_lag5_train, ipt_lag10_train], 3)

  # 検証データの作成
  idx = np.where((rt.year > 2015))[0]
  ipt_lag0_test = ipt_lag0[idx]
  ipt_lag5_test = ipt_lag5[idx]
  ipt_lag10_test = ipt_lag10[idx]
  #ipt_test = ipt[idx]
  #ipt_test = np.concatenate([ipt_lag0_test, ipt_lag5_test, ipt_lag10_test], 1)
  ipt_test = np.stack([ipt_lag0_test, ipt_lag5_test, ipt_lag10_test], 3)
  return ipt_train, ipt_test

# === main routine ===
lead_time = 0
data, rt, sup_train, sup_test, output_shape, rt_test = indexing(lead_time) 
olr_ipt_train, olr_ipt_test = preprocess(olr_norm, rt, lead_time)
u850_ipt_train, u850_ipt_test = preprocess(u850_norm, rt, lead_time)
#v850_ipt_train, v850_ipt_test = preprocess(v850_norm, rt, lead_time)
u200_ipt_train, u200_ipt_test = preprocess(u200_norm, rt, lead_time)
#v200_ipt_train, v200_ipt_test = preprocess(v200_norm, rt, lead_time)
h850_ipt_train, h850_ipt_test = preprocess(h850_norm, rt, lead_time)
pr_wtr_ipt_train, pr_wtr_ipt_test = preprocess(pr_wtr_norm, rt, lead_time)

ipt_train = np.concatenate([olr_ipt_train, u850_ipt_train,  u200_ipt_train,
#                            v850_ipt_train, v200_ipt_train, 
                            h850_ipt_train, pr_wtr_ipt_train], 3)
ipt_test  = np.concatenate([olr_ipt_test, u850_ipt_test,  u200_ipt_test,
#                            #v850_ipt_test, v200_ipt_test, 
                            h850_ipt_test, pr_wtr_ipt_test], 3)
#ipt_train = olr_ipt_train
#ipt_test = olr_ipt_test
print(ipt_train.shape, ipt_test.shape)

model_path = '/home/maeda/machine_learning/results/model/kikuchi-7vals_v1/olr-u-h850-prw/model_7vals_0day.hdf5'
model = load_model(model_path)

print('===== Culicurating Gradient =====')
month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for mm in range(12):
    month_idx = np.where((rt_test.month == mm+1))[0]
    ipt_test_mm = ipt_test[month_idx]
    number = ipt_test_mm.shape[0]
    grads = np.zeros((number, 25, 144, 3*5))
    for num in range(number):
        img = keras.preprocessing.image.img_to_array(ipt_test[num])
        img = img.reshape((1, *img.shape))
        y_pred = model.predict(img)   # 入力した img に対する推論結果（PC1, PC2）

        images = tf.Variable(img, dtype=float)  # 9 channel の images(物理量)

        with tf.GradientTape() as tape:
            pred = model(images, training=False)  # 入力した img に対する推論結果（PC1, PC2)
            loss = tf.keras.losses.mean_squared_error(sup_test[num,:], pred[:,:])  # PC1, PC2 の指定を行うこと!!!

        grads[num] = tape.gradient(loss, images)   # dy_dx = tape.gradient(y, x)
        
    print('month = ', month[mm])
    #print('(PC1, PC2) = ', sup_test[num])
    print('shape = ', grads.shape)
    np.savez('/home/maeda/machine_learning/results/kikuchi-7vals_v1/saliency-map/5vals/grads_0day_' + str(month[mm]) + '.npz', grads=grads)

#print('===== Drawing Pictures =====')
#grad_std = grads.std(axis=0)
#name_box = ['OLR', 'U850', 'U200', 'H850', 'Precipitable Water']
#lag_box = ['day-0', 'day-5', 'day-10']
#for jj in range(5):
#    fig = plt.figure(figsize=(12,5))
#    for cc in range(3):
#        ax=fig.add_subplot(3,1,cc+1, projection=ccrs.PlateCarree(central_longitude=180))
#        # 上下の余白を調整
#        plt.subplots_adjust(hspace=0)
#        ax.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())
#        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0)
#        gl.top_labels = False     # 上部の経度のラベルを消去
#        if cc != 2:
#            gl.bottom_labels = False
#        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 30)) # 経度線
#        gl.ylocator = mticker.FixedLocator(np.arange(-30, 30, 10)) # 緯度線
#        ax.coastlines(color='gray')
#
#        lon = np.linspace(0, 360, 144)
#        lat = np.linspace(30, -30, 25)
#        x, y = np.meshgrid(lon-180.0, lat) # 経度、緯度データ
#        cntr = ax.pcolormesh(x, y, grad_std[:,:,3*jj+cc], vmax=0.002, vmin=0, cmap='magma')
#        cbar = fig.colorbar(cntr,ticks = np.linspace(0, 0.002, 6), extend='both', orientation='vertical', shrink=0.9)
#        ax.text(140, 24, lag_box[cc-1], fontsize=16, ha='left', va='center', bbox=dict(facecolor='w', edgecolor='w', boxstyle='round,pad=0.1', alpha=0.8))
#        if cc == 0:
#            ax.set_title('Gradient Map  ' + str(name_box[jj]) + '   Lead Time = ' + str(lead_time) + 'days')
#        ax.axis((-180, 180, -30, 30))
#    plt.savefig('/home/maeda/machine_learning/results/kikuchi-7vals_v1/saliency-map/5vals/gradient-all_std_lt0_' + str(name_box[jj]) + '.png', 
#                bbox_inches='tight', pad_inches=0.1, dpi=200)    
  
print('===== FINISH =====')