import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from tensorflow import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

input_dir1 = '/home/maeda/data/geosciAI24/TC_data_GeoSciAI2024/'
input_dir2 = '/home/maeda/data/geosciAI24/TC_data_GeoSciAI2024_test/'
output_dir = '/home/maeda/machine_learning/results/'

ensemble = True
def get_input_ans(start_year, end_year, input_dir, n_input = 1):
    trackfiles = []
    field = ['olr', 'qv600', 'slp', 'sst', 'u200', 'u850', 'v200']
    FIELD = ['OLR', 'QV600', 'SLP', 'SST', 'U200', 'U850', 'V200']
    for i in range(start_year, end_year+1):
        trackfiles += glob.glob(input_dir + f'track_data/{i}*.csv')

    input = []
    ans = []
    times = []
    init = []
    for file in trackfiles:
        df = pd.read_csv(file)
        col = df.columns
        colname = col[6] #TCフラグ
        tc_df = df[df[colname] == 1] #TCフラグが１のところだけ抽出
        index = tc_df.index
        start_index = index[0]
        end_index = min([index[-1], df.shape[0]-5]) #最後のTCフラグ1の時点から24時間後（4ステップ先）の予測をしたいがデータがないかもしれない。

        time = np.array(df.iloc[start_index:end_index+4+1, 0])
        lon  = np.array(df.iloc[start_index:end_index+4+1, 1])
        lat  = np.array(df.iloc[start_index:end_index+4+1, 2])
        wind = np.array(df.iloc[start_index:end_index+4+1, 3])
        tsteps = wind.shape[0] - 4
        # SST data にあわせるため、初期は 00 時刻のデータを読み込む
        if time[0][:2] == '06':
            time = time[3:]
            lon  = lon[3:]
            lat  = lat[3:]
            wind = wind[3:]
            tsteps = tsteps - 3 
        elif time[0][:2] == '12':
            time = time[2:]
            lon  = lon[2:]
            lat  = lat[2:]
            wind = wind[2:]
            tsteps = tsteps - 2
        elif time[0][:2] == '18':
            time = time[1:]
            lon  = lon[1:]
            lat  = lat[1:]
            wind = wind[1:] 
            tsteps = tsteps - 1  

        # 予測対象の画像データを取得
        for ii in range(tsteps):
            x = np.zeros((64, 64, len(field)), dtype=np.float32)
            if lat[ii]<0:
                ns = "s"
            else:
                ns = "n"
            # 四捨五入を１０進数で正確に行う
            round_lon = Decimal(lon[ii]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
            round_lat = Decimal(lat[ii]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
            if round_lon == 360:
                round_lon = 0
            ymdh = datetime.strptime(time[ii], "%HZ%d%b%Y").strftime('%Y%m%d%H')
            if str(ymdh[-2:]) == '00':   # SST は daily data なので00時の位置データで読み込む
                ymdh_sst = datetime.strptime(time[ii], "%HZ%d%b%Y").strftime('%Y%m%d%H')
                lon_sst = lon[ii]
                lat_sst = lat[ii]
                if lat_sst<0:
                    ns_sst = "s"
                else:
                    ns_sst = "n"
            # 対応するデータの読み込み
            for jj in range(len(field)):
                if field[jj] == 'sst':   # SST の読み込み
                    round_lon_sst = Decimal(lon_sst).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
                    round_lat_sst = Decimal(lat_sst).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
                    if round_lon_sst == 360:
                        round_lon_sst = 0
                    filename = f"{field[jj]}_{ymdh_sst}_{(round_lon_sst):03}_{(abs(round_lat_sst)):03}{ns_sst}.npz"
                    xi = np.load(input_dir + f'field_data/{ymdh_sst[:4]}/{FIELD[jj]}/{filename}')
                else:
                    filename = f"{field[jj]}_{ymdh}_{(round_lon):03}_{(abs(round_lat)):03}{ns}.npz"
                    xi = np.load(input_dir + f'field_data/{ymdh[:4]}/{FIELD[jj]}/{filename}')
                x[:,:,jj] = xi['data']
                
            input.append(x)
            ans.append(wind[ii+n_input+4-1])
            init.append(wind[ii+n_input-1])
            times.append(time[ii])
    return np.array(input), np.array(ans), np.array(times), np.array(init)
'''''
def get_input_ans(start_year, end_year, input_dir, n_input = 1):
    trackfiles = []
    field = ['olr', 'qv600', 'slp', 'u200', 'u850', 'v200', 'v850']
    FIELD = ['OLR', 'QV600', 'SLP', 'U200', 'U850', 'V200', 'V850']
    #field = ['olr']
    #FIELD = ['OLR']
    for i in range(start_year, end_year+1):
        trackfiles += glob.glob(input_dir + f'track_data/{i}*.csv')

    input = []
    ans   = []
    times = []
    for file in trackfiles:
        df = pd.read_csv(file)
        col = df.columns
        colname = col[6] #TCフラグ
        tc_df = df[df[colname] == 1] #TCフラグが１のところだけ抽出
        index = tc_df.index
        start_index = index[0] #発生時に２ステップ前のデータを使いたいが、ない場合は諦める。
        end_index = min([index[-1], df.shape[0]-5]) #最後のTCフラグ1の時点から24時間後（4ステップ先）の予測をしたいがデータがないかもしれない。

        time = np.array(df.iloc[start_index:end_index+4+1, 0])
        lon  = np.array(df.iloc[start_index:end_index+4+1, 1])
        lat  = np.array(df.iloc[start_index:end_index+4+1, 2])
        wind = np.array(df.iloc[start_index:end_index+4+1, 3])
        tsteps = wind.shape[0] - 4
        
        # 予測対象の画像データを取得
        for ii in range(tsteps):
            x = np.zeros((64, 64, len(field)), dtype=np.float32)
            if lat[ii]<0:
                ns = "s"
            else:
                ns = "n"
            # 四捨五入を１０進数で正確に行う
            round_lon = Decimal(lon[ii]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
            round_lat = Decimal(lat[ii]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
            if round_lon == 360:
                round_lon = 0
            ymdh = datetime.strptime(time[ii], "%HZ%d%b%Y").strftime('%Y%m%d%H')
            # 対応するデータの読み込み
            for jj in range(len(field)):
                #if jj == 3:   # SST は daily data なので時間を丸めて処理
                #    ymdh = datetime.strptime(time[ii], "%HZ%d%b%Y").strftime('%Y%m%d00')
                filename = f"{field[jj]}_{ymdh}_{(round_lon):03}_{(abs(round_lat)):03}{ns}.npz"
                xi = np.load(input_dir + f'field_data/{ymdh[:4]}/{FIELD[jj]}/{filename}')
                x[:,:,jj] = xi['data']
                
            times.append(time[ii])
            input.append(x)
            ans.append(wind[ii+n_input+4-1])
            
    return np.array(input), np.array(ans), np.array(times)
'''''
# CNNモデルの構築
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (2, 2), padding='same', input_shape=(64, 64, 7), strides=(2,2), kernel_regularizer=l2(0.001)))   
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    model.add(Dropout(0.1))                                                                            
    model.add(Conv2D(64, (2, 2), padding='same', strides=(2,2)))                                        
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))                      
    model.add(Conv2D(128, (2, 2), padding='same', strides=(2,2)))                           
    model.add(BatchNormalization())                          
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Flatten())  # 一次元の配列に変換                          
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()
    return model

# 学習曲線の描画
def learning_curve(history, output_dir, seed, ensemble):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')    #Validation loss : 精度検証データにおける損失
    plt.xlim(0, 50)
    plt.ylim(0, 1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    if ensemble == True:
        plt.savefig(output_dir + f'geosciAI24/l_curve/predict8-{(seed):03}.png')
    else:
        plt.savefig(output_dir + 'geosciAI24/l_curve/predict8.png')
    plt.close()


if __name__ == "__main__": 
    print('Loading Data...')
    input_train, ans_train, times_train, init_train = get_input_ans(1979, 1999, input_dir1)
    input_valid, ans_valid, times_valid, init_valid = get_input_ans(2000, 2003, input_dir1)
    input_test,  ans_test,  times_test,  init_test  = get_input_ans(2004, 2009, input_dir2)
    print('input_train, ans_train = ', input_train.shape, ans_train.shape)
    print('input_valid, ans_valid = ', input_valid.shape, ans_valid.shape)
    print('input_test,  ans_test = ', input_test.shape, ans_test.shape)
    
    # 標準化処理
    print('Normalization...')
    ipt = np.concatenate([input_train, input_valid], axis=0)
    ans = np.concatenate([ans_train, ans_valid], axis=0)
    input_std  = np.nanstd(ipt, axis=0)
    input_mean = np.nanmean(ipt, axis=0)
    ans_std    = np.nanstd(ans, axis=0)
    ans_mean   = np.nanmean(ans, axis=0)

    input_train = (input_train - input_mean) / input_std
    input_valid = (input_valid - input_mean) / input_std
    input_test  = (input_test - input_mean) / input_std
    ans_train   = (ans_train - ans_mean) / ans_std
    ans_valid   = (ans_valid - ans_mean) / ans_std
    ans_test    = (ans_test - ans_mean) / ans_std
    # 欠損値のゼロ埋め
    input_train = np.nan_to_num(input_train, nan=0)
    input_valid = np.nan_to_num(input_valid, nan=0)
    input_test  = np.nan_to_num(input_test, nan=0)
    print('ans_mean, ans_std = ', ans_mean, ans_std)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    if ensemble == True:
        scores = []
        predicts = []
        seeds = [7,8,9,10,12,13,14,15]
        
        for seed in range(36, 37):
            print('Seed = ', seed)
            random.set_seed(seed)  # TensorFlowのseed値を設定
            np.random.seed(seed)  
            # モデルの構築とコンパイル
            model = cnn_model()
            callback = EarlyStopping(monitor='loss',patience=3)
            model.compile(optimizer=Adam(), loss='mean_squared_error')
            history = model.fit(input_train, ans_train, epochs=30, batch_size=64, 
                                validation_data=(input_valid, ans_valid), 
                                callbacks=[callback])
            
            learning_curve(history, output_dir, seed, ensemble)
            # 評価データによるテスト
            score = model.evaluate(input_test, ans_test)
            predict = model.predict(input_test, batch_size=None, verbose=0, steps=None) # モデルの出力を獲得する
            print('predict = ', predict.shape)
            # 標準化を元に戻す
            predict = predict * ans_std + ans_mean
            # モデルデータの保存
            model.save(output_dir + f'/model/model8_test{(seed):03}_wo-v850.h5')
            # 評価データの保存
            np.savez(output_dir + f'geosciAI24/predict/predict8_test{(seed):03}_wo-v850.npz', 
                    predict=predict, ans=ans_test, 
                    history=history.history, score=score, time=times_test, init_wind=init_test)
            
            scores.append(score)
            predicts.append(predict)
        
        scores = np.array(scores)
        predicts = np.array(predicts)
        print('scores = ', scores)
        print('mean score, std score = ', np.mean(scores), np.std(scores))
        
        np.savez(output_dir + 'geosciAI24/predict/predict_ensemble8.npz', 
                    seeds=seeds, predict=predicts, ans=ans_test, score=scores)
    
    else:
        seed = None
        
        model = cnn_model()
        callback = EarlyStopping(monitor='loss',patience=3)
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        history = model.fit(input_train, ans_train, epochs=30, batch_size=64, 
                            validation_data=(input_valid, ans_valid), 
                            callbacks=[callback])
        
        learning_curve(history, output_dir, seed, ensemble)
        score = model.evaluate(input_test, ans_test)
        predict = model.predict(input_test, batch_size=None, verbose=0, steps=None) # モデルの出力を獲得する
        print('predict = ', predict.shape)
        predict = predict * ans_std + ans_mean
        
        model.save(output_dir + f'/model/model_wo-olr.h5')
        np.savez(output_dir + f'geosciAI24/predict/predict_wo-u850.npz', 
                predict=predict, ans=ans_test, 
                history=history.history, score=score, init_wind=init_test)
        
