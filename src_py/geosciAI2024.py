import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, BatchNormalization, LayerNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

input_dir = '/home/maeda/data/geosciAI24/TC_data_GeoSciAI2024/'

def get_input_ans(start_year, end_year, n_input = 1):
    trackfiles = []
    field = ['olr', 'qv600', 'slp', 'u200', 'v200']
    FIELD = ['OLR', 'QV600', 'SLP', 'U200', 'V200']
    #field = ['olr']
    #FIELD = ['OLR']
    for i in range(start_year, end_year+1):
        trackfiles += glob.glob(input_dir + f'track_data/{i}*.csv')

    input=[]
    ans =[]
    for file in trackfiles:
        df = pd.read_csv(file)
        col = df.columns
        colname = col[6] #TCフラグ
        tc_df = df[df[colname] == 1] #TCフラグが１のところだけ抽出
        index = tc_df.index
        start_index = max([index[0], 2]) #発生時に２ステップ前のデータを使いたいが、ない場合は諦める。
        end_index = min([index[-1], df.shape[0]-5]) #最後のTCフラグ1の時点から24時間後（4ステップ先）の予測をしたいがデータがないかもしれない。

        time = np.array(df.iloc[start_index-2:end_index+4+1, 0])
        lon  = np.array(df.iloc[start_index-2:end_index+4+1, 1])
        lat  = np.array(df.iloc[start_index-2:end_index+4+1, 2])
        wind = np.array(df.iloc[start_index-2:end_index+4+1, 3])
        tsteps = wind.shape[0] - 2 - 4
        
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
                
            input.append(x)
            ans.append(wind[ii+n_input+4-1])
    return np.array(input), np.array(ans)

print('Loading Data...')
input_train, ans_train = get_input_ans(1979, 1999)
input_valid, ans_valid = get_input_ans(2000, 2003)

# 欠損値のゼロ埋め
input_train = np.nan_to_num(input_train, nan=0)
input_valid = np.nan_to_num(input_valid, nan=0)
ans_train   = np.nan_to_num(ans_train, nan=0)
ans_valid   = np.nan_to_num(ans_valid, nan=0)

print('input_train = ',input_train.shape)
print('ans_train   = ', ans_train.shape)
print('input_valid = ', input_valid.shape)
print('ans_valid   = ', ans_valid.shape)

# 標準化処理
print('Normalization...')
input_std  = np.mean(input_train, axis=0)
input_mean = np.mean(input_train, axis=0)
ans_std    = np.std(ans_train, axis=0)
ans_mean   = np.std(ans_train, axis=0)

input_train = (input_train - input_mean) / input_std
input_valid = (input_valid - input_mean) / input_std
ans_train   = (ans_train - ans_mean) / ans_std
ans_valid   = (ans_valid - ans_mean) / ans_std

# CNNモデルの構築
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (2, 2), padding='same', input_shape=(64, 64, 5), strides=(2,2)))   
    model.add(BatchNormalization())
    #model.add(LayerNormalization())
    model.add(Activation('relu'))                                           
    model.add(Conv2D(64, (2, 2), padding='same', strides=(2,2)))                                        
    model.add(BatchNormalization())
    #model.add(LayerNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (2, 2), padding='same', strides=(2,2)))                           
    model.add(BatchNormalization())
    #model.add(LayerNormalization())
    #model.add(Conv2D(256, (2, 2), padding='same', strides=(2,2)))                           
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2)) 

    model.add(Flatten())  # 一次元の配列に変換                          
    model.add(Dense(128))
    model.add(Activation('relu'))
    #model.add(Dense(64))
    model.add(Dense(1, activation='linear'))
    model.summary()
    return model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = cnn_model()
callback = EarlyStopping(monitor='loss',patience=3)
model.compile(optimizer=Adam(), loss='mean_squared_error')
history = model.fit(input_train, ans_train, epochs=100, batch_size=128, validation_data=(input_valid, ans_valid), callbacks=[callback])
