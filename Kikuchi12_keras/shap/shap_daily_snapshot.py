import numpy as np
from numpy.linalg.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.markers import MarkerStyle
import cartopy.crs as ccrs
import cartopy.feature as cfeaturel
import pandas as pd


def load_file(lead_time, seed, lt_max):
    df  = np.load(f'/work/gi55/i55233/data/results/bsiso_shap/torch_{lead_time:03}day{seed:03}-all.npz')
    df2 = np.load(f'/work/gi55/i55233/data/machine_learning/results/kikuchi-single/cor/8vals/{lead_time:03}day/torch_seed{seed:03}.npz')
    df3 = np.load(f'/work/gi55/i55233/data/machine_learning/results/kikuchi-single/cor/8vals/000day/torch_seed{seed:03}.npz')
    shap_data = df['shap_values']
    pred = df2['arr_0'][:-1-(lt_max-lead_time)] 
    initial = df3['arr_0'][:-1-(lt_max)] 
    rt = pd.to_datetime(df['time'], format='%Y%m%d')
    del df, df2, df3
    return shap_data, pred, initial, rt


def jja_indexing(rt, shap_data):
    jja = (rt.month>=6) & (rt.month<=8)
    shap_jja = shap_data[:,jja]
    rt_jja = rt[jja]
    return shap_jja, rt_jja


# 解像度を(2, 8, 25, 144) -> (2, 8, 13, 72) に変更する
def reduce_resolution(arr):     
    new_shape = (2, arr.shape[1], 8, 13, 72)
    new_arr = np.zeros(new_shape)
    for i in range(13):
        for j in range(72):
            new_arr[:,:,:,i,j] = np.sum(arr[:,:,:,i*2:i*2+2,j*2:j*2+2], axis=(3,4))
    return new_arr


# daily plot
def plot_daily_shap(lead_time, rt_jja, shap_jja, tt):
    fig = plt.figure(figsize=(15,15))
    for pcs in range(2):  
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        for jj in range(8):
            ax=fig.add_subplot(8,2,(jj*2)+1+pcs, projection=ccrs.PlateCarree(central_longitude=180))
            ax.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
            gl.top_labels = False     # 上部の経度のラベルを消去
            if pcs != 0:
                gl.left_labels = False    # 左側の経度のラベルを消去
            if jj != 7:
                gl.bottom_labels = False  # 下部の緯度のラベルを消去
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 30)) # 経度線
            gl.ylocator = mticker.FixedLocator(np.arange(-30, 30, 10)) # 緯度線
            ax.coastlines()

            lon = np.linspace(0, 360, 72)
            lat = np.linspace(30, -30, 13)
            x, y = np.meshgrid(lon-180.0, lat) # 経度、緯度データ
            #cntr = ax.pcolormesh(x, y, a[jj,:, :, cc], vmax=0.0005, vmin=-0.0005, cmap='RdBu_r')
            cntr = ax.contourf(x, y, shap_jja[pcs,tt,jj,:,:], np.linspace(-0.01, 0.01), cmap='RdBu_r', extend='both')
            #cbar = fig.colorbar(cntr, ticks = np.linspace(-0.0005, 0.0005, 6), orientation='vertical')
            #ax.set_title('Shaprey Value  ' + str(name_box[jj]))
            ax.axis((-179, 60, -30, 30))

    cbar = fig.add_axes([0.92, 0.25, 0.015, 0.5]) # 順に[左からの位置、下からの位置、横幅、縦幅]
    fig.colorbar(cntr, ticks=np.linspace(-0.01, 0.01, 6), orientation='vertical', cax=cbar)
    plt.savefig(f'/work/gi55/i55233/data/results/bsiso_shap/jja_daily_cmap/{lead_time:03}/{rt_jja[tt].strftime("%Y-%m-%d")}.png', bbox_inches='tight', dpi=150)
    plt.close()
    return



if __name__ == '__main__':
    seed = 0
    lt_max = 30
    lead_time = 15
    name_box = ['OLR', 'U850', 'U200', 'V850', 'V200', 'H850', 'PW', 'SST']
    
    shap_data, pred, initial, rt = load_file(lead_time, seed, lt_max)
    shap_data = reduce_resolution(shap_data)
    shap_jja, rt_jja = jja_indexing(rt, shap_data)
    
    for tt in range(90):
        plot_daily_shap(lead_time, rt_jja, shap_jja, tt)
        if tt%10 == 0:
            print(tt)

