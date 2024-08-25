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
    pred = df2['arr_0'][tt:-1-lt_max] # 対応する pred の値
    initial = df3['arr_0'][:-1-lt_max] # 初期値
    # shap の絶対値の寄与度を計算
    #shap_abs = np.abs(shap_data.sum(axis=(2,3)))
    shap_abs = shap_data.sum(axis=(2,3))
    
    shap_val_abs = np.zeros((shap_abs.shape[0], shap_abs.shape[1], 8))
    for j in range(8):
        shap_val_abs[:,:,j] = shap_abs[:,:,j*3:(j+1)*3].sum(axis=2)
    
    print(shap_val_abs.shape) # (2, 644, 8)
    
    # 20150101-20231231 のデータを取得
    sup_rt = pd.date_range('2016-01-01', '2022-12-31', freq='D')
    sup_rt_jja = sup_rt[np.where((sup_rt.month >= 6) & (sup_rt.month <= 8))]
    for yy in range(2016, 2023):
        yy_jja = np.where((sup_rt.year == yy) & (sup_rt.month >= 6) & (sup_rt.month <= 8))[0]    
        pred_jja = pred[yy_jja]
        init_jja = initial[yy_jja]
        rt_yy = sup_rt[yy_jja]
        #print(rt_yy.shape, pred_jja.shape, init_jja.shape, sup_rt.shape)
        # shap のjja は要素数が異なるため別にインデクシングする
        jja = np.where((sup_rt_jja.year == yy))[0]
        
        
        # 3 * 3 のフェーズ分布を描画
        val = ['OLR', 'U850', 'U200', 'V850', 'V200', 'H850', 'PW', 'SST']
        fig = plt.figure(figsize=(16,6))
        ax1 = fig.add_subplot(2, 1, 1)
        # 高さの占める割合を調整する
        ax1.set_aspect(0.8)
        ax1.plot(rt_yy, pred_jja[:,pcs], color='g')
        ax1.set_xlim(rt_yy[0], rt_yy[-1])
        ax1.set_ylim(-3, 3)
        #ax1.set_yticks(np.arange(-3, 3.1, -1.5))
        ax1.hlines(0, rt_yy[0], rt_yy[-1], 'k', linestyle='dashed', linewidth=0.3)
        ax2 = fig.add_subplot(2, 1, 2)   
        # アスペクト比を 4:1
        #ax2.set_aspect(4)
        xi, yi = np.meshgrid(np.arange(92), np.arange(8))
        pcm = ax2.pcolormesh(xi, yi, shap_val_abs[pcs,jja].T, cmap='seismic', vmin=-0.6, vmax=0.6)

        ax2.set_xlabel('Time', fontsize=15)

        ax2.set_xlim(0, 91)
        ax2.set_ylim(7.5, -0.5)
        # y軸に text を表示
        ax2.set_yticks(np.arange(8))
        ax2.set_yticklabels(val, fontsize=15)
        # plot の幅を揃える
        fig.canvas.draw()
        axpos1 = ax1.get_position() # 上の図の描画領域
        axpos2 = ax2.get_position() # 下の図の描画領域
        #幅をax1と同じにする
        ax2.set_position([axpos2.x0, axpos2.y0, axpos1.width, axpos2.height+0.12]) # [左からの位置, 下からの位置, 幅, 高さ]
        cbar_ax = fig.add_axes([0.925, 0.12, 0.015, 0.43]) # [左からの位置, 下からの位置, 幅, 高さ]
        plt.colorbar(pcm, cax=cbar_ax)
        title = f'PC{pcs+1} SHAP values (Lead time = {tt} day)      year = {yy}'
        fig.suptitle(title, fontsize=20, y=0.83)
        # save
        plt.savefig(f'/work/gi55/i55233/data/results/bsiso_shap/heatmap_tmseries/{(tt):03}/pc{pcs+1}_{yy}.png', bbox_inches='tight', dpi=200)
        plt.close()

if __name__ == '__main__':
    
    for tt in [5, 10, 15, 20, 25, 30]:
        for pcs in range(2):
            main(tt, pcs)
    
        
    