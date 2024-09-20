import numpy as np
from numpy.linalg.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.markers import MarkerStyle
import cartopy.crs as ccrs
import cartopy.feature as cfeaturel
import pandas as pd
#import shap

seed = 10
lt_max = 30
lead_time = 30

df  = np.load(f'/work/gi55/i55233/data/results/bsiso_shap/torch_{lead_time:03}day.npz')
df2 = np.load(f'/work/gi55/i55233/data/machine_learning/results/kikuchi-single/cor/8vals/{lead_time:03}day/torch_seed000.npz')
df3 = np.load(f'/work/gi55/i55233/data/machine_learning/results/kikuchi-single/cor/8vals/000day/torch_seed000.npz')
shap_data = df['shap_values']
pred = df2['arr_0'][:-1-(lt_max-lead_time)] # 対応する pred の値
initial = df3['arr_1'][:-1-(lt_max)] # 初期値の正解データ

print(shap_data.shape)
print(pred.shape)
print(initial.shape)


rt = pd.to_datetime(df['time'], format='%Y%m%d')
jja = (rt.month>=6) & (rt.month<=8)
rt2 = pd.date_range(start=rt[0], periods=len(pred), freq='D')
jja2 = (rt2.month>=6) & (rt2.month<=8)

shap_values = shap_data.sum(axis=(3, 4))
shap_abs = np.abs(shap_data).sum(axis=(3, 4))
shap_jja = shap_data[:,:]
rt_jja = rt[jja]
initial_jja = initial[jja2]
pred_jja = pred[jja2]

print(shap_values[:,0,:])
print(shap_values.shape, pred.shape, shap_abs.shape)

# Amp >= 1.0 の場合のみを抽出
amp = np.sqrt(initial_jja[:,0]**2 + initial_jja[:,1]**2)
amp_strong = np.where(amp>=1.)[0]

shap_values_strong = shap_values[:,amp_strong] 
print(shap_values_strong.shape, shap_values.shape, pred.shape)

# 解像度を(2, 25, 144, 24) -> (2, 13, 72, 8) に変更する
def reduce_resolution(arr):
    new_shape = (2, arr.shape[1], 8, 13, 72)
    new_arr = np.zeros(new_shape)

    for i in range(13):
        for j in range(72):
            new_arr[:,:,:,i,j] = np.sum(arr[:,:,:,i*2:i*2+2,j*2:j*2+2], axis=(3,4))
    return new_arr

# 解像度を下げる
print(shap_jja.shape)
shap_jja = reduce_resolution(shap_jja)
print(shap_jja.shape)

# initial phase analysis
phase = np.zeros(initial_jja[:,0].shape, dtype=int)
# arctan2 ：
arg = np.arctan2(-initial_jja[:,1], -initial_jja[:,0]) # pc2/pc1
amp = np.sqrt(initial_jja[:,0]**2 + initial_jja[:,1]**2)
print(arg[:30])

# 8phase に分ける
phase[(arg >= -np.pi) & (arg < -3*np.pi/4)] = 1
phase[(arg >= -3*np.pi/4) & (arg < -np.pi/2)] = 2
phase[(arg >= -np.pi/2) & (arg < -np.pi/4)] = 3
phase[(arg >= -np.pi/4) & (arg < 0)] = 4
phase[(arg >= 0) & (arg < np.pi/4)] = 5
phase[(arg >= np.pi/4) & (arg < np.pi/2)] = 6
phase[(arg >= np.pi/2) & (arg < 3*np.pi/4)] = 7
phase[(arg >= 3*np.pi/4) & (arg <= np.pi)] = 8
#


shap_jja_strong = shap_jja[:,amp_strong] 
pred_jja_strong = pred_jja[amp_strong]

# plot
name_box = ['OLR', 'U850', 'V850', 'U200', 'V200', 'H850', 'PW', 'SST']
print('lead time = ',lead_time)
for pcs in range(2):  
    print('PC = ', pcs+1)
    fig = plt.figure(figsize=(60,20))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    for ch in range(8):
        for jj in range(8):
            ph = np.where(phase == jj+1)[0]
            ax=fig.add_subplot(8,8,jj*8+ch+1, projection=ccrs.PlateCarree(central_longitude=180))
            ax.set_aspect(1.2)
            ax.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
            gl.top_labels = False     # 上部の経度のラベルを消去
            if jj != 7:
                gl.bottom_labels = False  # 下部の緯度のラベルを消去
            if ch != 0:
                gl.left_labels = False
            
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 30)) # 経度線
            gl.ylocator = mticker.FixedLocator(np.arange(-30, 30, 10)) # 緯度線
            ax.coastlines()

            lon = np.linspace(0, 360, 72)
            lat = np.linspace(30, -30, 13)
            x, y = np.meshgrid(lon-180.0, lat) # 経度、緯度データ
            cntr = ax.pcolormesh(x, y, shap_jja[pcs,ph,ch,:,:].mean(axis=0), vmax=0.005, vmin=-0.005, cmap='bwr')
            if jj == 0:
                ax.text(-175, 18, name_box[ch], fontsize=24, fontweight='bold')
            cbar_ax = fig.add_axes([0.91, 0.35, 0.005, 0.3]) # [左からの位置, 下からの位置, 幅, 高さ]
            #cntr = ax.contourf(x, y, shap_jja[0,ph,7,:,:].mean(axis=0), np.linspace(-0.01, 0.01), cmap='RdBu_r', extend='both')
            cbar = fig.colorbar(cntr, ticks = np.linspace(-0.005, 0.005, 6), orientation='vertical', cax=cbar_ax)
            #ax.set_title('Shaprey Value  ' + str(name_box[jj]))
            ax.axis((-179, 60, -30, 30))
        #cbar = fig.colorbar(cntr, ticks = np.linspace(-0.0005, 0.0005, 6), orientation='horizontal')
    plt.savefig(f'/work/gi55/i55233/data/results/bsiso_shap/phase_comp/{lead_time:03}_PC{pcs+1}.png', bbox_inches='tight', dpi=300)
    plt.close()