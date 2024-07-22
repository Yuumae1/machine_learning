import numpy as np
from numpy.linalg.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.markers import MarkerStyle
import cartopy.crs as ccrs
import cartopy.feature as cfeaturel
import pandas as pd

seed = [ 8, 15, 16,  0, 19,  0,  0, 19, 12,  7,
         1, 19, 18, 17, 17,  2,  7, 16, 18, 16,
        16, 15, 16, 16, 12, 13, 19, 19, 18, 18, 
        18, 13, 15, 18, 18, 18]


def main(tt, pcs):
    print('tt, pcs = ', tt, pcs)
    lt_max = len(seed)
    df  = np.load(f'/work/gi55/i55233/data/results/bsiso_shap/jja_{(tt):03}day.npz')
    df2 = np.load(f'/work/gi55/i55233/data/machine_learning/results/kikuchi-8vals_v1/cor/8vals/{(tt):03}day/seed{(seed[tt]):03}.npz')
    df3 = np.load(f'/work/gi55/i55233/data/machine_learning/results/kikuchi-8vals_v1/cor/8vals/000day/seed{(seed[0]):03}.npz')

    shap_data = df['shap_values']
    pred = df2['arr_0'][:-1-(lt_max-tt)] # 対応する pred の値
    initial = df3['arr_0'][:-1-(lt_max)] # 初期値
    #print(shap_data.shape)
    #print(pred.shape)
    #print(initial.shape)
    
    # shap の絶対値の寄与度を計算
    shap_abs = np.abs(shap_data.sum(axis=(2,3)))
    
    shap_val_abs = np.zeros((shap_abs.shape[0], shap_abs.shape[1], 8))
    for j in range(8):
        shap_val_abs[:,:,j] = shap_abs[:,:,j*3:(j+1)*3].sum(axis=2)
    
    
    # 20150101-20231231 のデータを取得
    sup_rt = pd.date_range('2016-01-01', '2022-12-31', freq='D')
    jja = np.where((sup_rt.year > 2015) & (sup_rt.month >= 6) & (sup_rt.month <= 8))[0]    
    pred_jja = pred[jja]
    init_jja = initial[jja]
    #print(pred.shape)
    #print(pred_jja.shape, init_jja.shape)
    
    # skill of phase representation
    bin_x = np.arange(-3, 3.1, 0.5)
    bin_y = np.arange(-3, 3.1, 0.5)

    #print(bin_x)
    # bin ごとの平均寄与度を計算
    shap_bin = np.zeros((len(bin_x)-1, len(bin_y)-1, 8))
    for i in range(len(bin_x)-1):
        for j in range(len(bin_y)-1):
            idx = np.where((pred_jja[:,0] >= bin_x[i]) & (pred_jja[:,0] < bin_x[i+1]) & (pred_jja[:,1] >= bin_y[j]) & (pred_jja[:,1] < bin_y[j+1]))[0]
            if len(idx) > 0:
                shap_bin[i,j] = shap_val_abs[pcs,idx].mean(axis=0)
            else:
                shap_bin[i,j] = np.nan
    
    cnt_rate = shap_bin / shap_bin.sum(axis=2)[:,:,np.newaxis]
    
    # 3 * 3 のフェーズ分布を描画
    val = ['OLR', 'U850', 'U200', 'V850', 'V200', 'H850', 'PW', 'SST']
    fig = plt.figure(figsize=(16,16))
    for i in range(8):
        ax = fig.add_subplot(3,3,i+1)
        # アスペクト比を 1:1 に
        ax.set_aspect('equal')
        # 間隔を開ける
        # r = 1 の円
        theta = np.linspace(0, 2*np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.plot(x, y, 'k', linewidth=0.5)
        # 中心線
        ax.hlines(0, -3, 3, 'k', linewidth=0.5)
        ax.vlines(0, -3, 3, 'k', linewidth=0.5)
        # 斜線
        ax.plot([-3, -np.sqrt(2)/2], [-3, -np.sqrt(2)/2], 'k', linewidth=0.5)
        ax.plot([np.sqrt(2)/2, 3], [np.sqrt(2)/2, 3], 'k', linewidth=0.5)
        ax.plot([-3, -np.sqrt(2)/2], [3, np.sqrt(2)/2], 'k', linewidth=0.5)
        ax.plot([np.sqrt(2)/2, 3], [-np.sqrt(2)/2, -3], 'k', linewidth=0.5)

        xi, yi = np.meshgrid(bin_x, bin_y)
        pcm = ax.pcolormesh(xi, yi, cnt_rate[:,:,i], cmap='YlOrRd', vmin=0, vmax=0.3)

        ax.set_xlabel('-PC1', fontsize=15)
        ax.set_ylabel('-PC2', fontsize=15)
        ax.set_xticks(np.arange(3, -3, -1))
        ax.set_yticks(np.arange(3, -3, -1))
        ax.text(-2.5, 2.5, val[i], fontsize=15)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    cbar_ax = fig.add_axes([0.925, 0.3, 0.02, 0.4]) # [左からの位置, 下からの位置, 幅, 高さ]
    fig.colorbar(pcm, cax=cbar_ax)  
    # save
    plt.savefig(f'/work/gi55/i55233/data/results/bsiso_shap/ph_cnt/phase_rep_{tt:03}_pc{pcs}.png', bbox_inches='tight')


if __name__ == '__main__':
    
    for tt in range(0,36):
        for pcs in range(2):
            main(tt, pcs)
    
        
    