import numpy as np
from numpy.linalg.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.markers import MarkerStyle
import cartopy.crs as ccrs
import cartopy.feature as cfeaturel
import pandas as pd
import shap

seed = [ 8, 15, 16,  0, 19,  0,  0, 19, 12,  7,
         1, 19, 18, 17, 17,  2,  7, 16, 18, 16,
        16, 15, 16, 16, 12, 13, 19, 19, 18, 18, 
        18, 13, 15, 18, 18, 18]
lt_max = len(seed)
tt = 25

df  = np.load(f'/work/gi55/i55233/data/results/bsiso_shap/jja_{(tt):03}day.npz')
df2 = np.load(f'/work/gi55/i55233/data/machine_learning/results/kikuchi-8vals_v1/cor/8vals/{(tt):03}day/seed{(seed[tt]):03}.npz')
shap_data = df['shap_values']
pred = df2['arr_0'][:-1-(lt_max-tt)] # 対応する pred の値

print(shap_data.shape)
print(pred.shape)

# データの読み込み

data = np.load('/work/gi55/i55233/data/results/bsiso_eeof/prepro_anomaly_8vals.npz')
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
data_file = '/work/gi55/i55233/data/results/bsiso_eeof/bsiso_rt-PCs.npz'
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
  idx = np.where((rt.year > 2015) & (rt.month == 6) & (rt.month == 7) & (rt.month == 8))[0]
  sup_test = sup_data[idx]
  sup_rt = rt[idx]
  print(sup_test.shape, sup_train.shape)
  return data, rt, sup_train, sup_test, output_shape, sup_rt

# 入力データの前処理
def preprocess(data, rt, lead_time):
  ipt_lag0  = data[10:-lead_time-1]
  ipt_lag5  = data[5:-lead_time-6]
  ipt_lag10 = data[:-lead_time-11]

  # 検証データの作成
  idx = np.where((rt.year > 2015) & ((rt.month == 6) | (rt.month == 7) | (rt.month == 8)))[0]
  ipt_lag0_test = ipt_lag0[idx]
  ipt_lag5_test = ipt_lag5[idx]
  ipt_lag10_test = ipt_lag10[idx]
  ipt_test = np.stack([ipt_lag0_test, ipt_lag5_test, ipt_lag10_test], 3)
  return ipt_test

# ==== iteration program ====
lt_box = [tt]
#lt_box = [0, 5, 10, 15, 20, 25, 30, 35]
#lt_box = np.arange(36)
for lead_time in lt_box:

  print('==== lead time : {} day ====='.format(lead_time))

  data, rt, sup_train, sup_test, output_shape, sup_rt = indexing(lead_time) 

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

# cor
# shap_data.shape = (2, 644, 25, 144, 24)
# ipt_test.shape = (644, 25, 144, 24)
eps = 1e-10
shap_cor = np.zeros((2, 25, 144, 24))
for pcs in range(2):
    for ch in range(24):
        for lt in range(25):
            for ln in range(144):
                shap_cor[pcs,lt,ln,ch] = np.corrcoef(shap_data[pcs,:,lt,ln,ch]+eps, ipt_test[:,lt,ln,ch]+eps)[0,1]
print(shap_cor)
# plot
name_box = ['OLR', 'U850', 'U200', 'V850', 'V200', 'H850', 'PW', 'SST']
print('lead time = ',tt)
t = 10

############################################
# shaprey value のプロット
for pcs in range(2):  
  fig = plt.figure(figsize=(28,20))
  plt.subplots_adjust(wspace=0.01, hspace=0.01)
  for jj in range(8):
    for cc in range(3):
      ax=fig.add_subplot(8,3,jj*3+(cc+1), projection=ccrs.PlateCarree(central_longitude=180))
      ax.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())
      gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
      gl.top_labels = False     # 上部の経度のラベルを消去
      if cc != 0:
        gl.left_labels = False    # 左側の経度のラベルを消去
      if jj != 7:
        gl.bottom_labels = False  # 下部の緯度のラベルを消去
      
      gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 30)) # 経度線
      gl.ylocator = mticker.FixedLocator(np.arange(-30, 30, 10)) # 緯度線
      ax.coastlines()

      lon = np.linspace(0, 360, 144)
      lat = np.linspace(30, -30, 25)
      x, y = np.meshgrid(lon-180.0, lat) # 経度、緯度データ
      #cntr = ax.pcolormesh(x, y, a[jj,:, :, cc], vmax=0.0005, vmin=-0.0005, cmap='RdBu_r')
      cntr = ax.contourf(x, y, shap_cor[pcs,:,:,jj*3+cc], np.linspace(-1, 1), cmap='RdBu_r', extend='both')
      #cbar = fig.colorbar(cntr, ticks = np.linspace(-0.0005, 0.0005, 6), orientation='vertical')
      #ax.set_title('Shaprey Value  ' + str(name_box[jj]))
      ax.axis((-179, 60, -30, 30))
  #cbar = fig.colorbar(cntr, ticks = np.linspace(-0.0005, 0.0005, 6), orientation='horizontal')
  plt.savefig(f'/work/gi55/i55233/data/results/bsiso_shap/bsiso_cor/{(tt):03}day_pcs{pcs}.png')
  plt.show()