import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

input_dir1 = '/home/maeda/data/geosciAI24/TC_data_GeoSciAI2024/'
input_dir2 = '/home/maeda/data/geosciAI24/TC_data_GeoSciAI2024_test/'
output_dir = '/home/maeda/machine_learning/results/'

ensemble = True

def get_input_ans(start_year, end_year, input_dir, n_input = 1):
    trackfiles = []
    field = ['olr', 'qv600', 'slp', 'u200', 'u850', 'v200', 'v850']
    FIELD = ['OLR', 'QV600', 'SLP', 'U200', 'U850', 'V200', 'V850']
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
                
            input.append(x)
            ans.append(wind[ii+n_input+4-1])
            
    return np.array(input), np.array(ans)

# CNNモデルの構築
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(7, 32, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*8*8, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x
    
model = CNNModel()


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
        plt.savefig(output_dir + f'geosciAI24/l_curve/predict{(seed):03}.png')
    else:
        plt.savefig(output_dir + 'geosciAI24/l_curve/predict.png')
    plt.close()


if __name__ == "__main__": 
    print('Loading Data...')
    input_train, ans_train = get_input_ans(1979, 1999, input_dir1)
    input_valid, ans_valid = get_input_ans(2000, 2003, input_dir1)
    input_test,  ans_test  = get_input_ans(2004, 2009, input_dir2)
    print('input_train, ans_train = ', input_train.shape, ans_train.shape)
    print('input_valid, ans_valid = ', input_valid.shape, ans_valid.shape)
    print('input_test,  ans_test = ', input_test.shape, ans_test.shape)
    
    # 標準化処理
    print('Normalization...')
    input_std  = np.nanstd(input_train, axis=0)
    input_mean = np.nanmean(input_train, axis=0)
    ans_std    = np.nanstd(ans_train, axis=0)
    ans_mean   = np.nanmean(ans_train, axis=0)

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
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        for seed in range(17, 30):
            print('Seed =', seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # モデルの初期化
            model = CNNModel()
            print(model)
            # データローダーの作成とモデルの訓練
            train_dataset = TensorDataset(torch.Tensor(input_train.transpose(0,3,1,2)), torch.Tensor(ans_train))
            train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            valid_dataset = TensorDataset(torch.Tensor(input_valid.transpose(0,3,1,2)), torch.Tensor(ans_valid))
            valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
            
            best_loss = float('inf')
            early_stopping_counter = 0
            
            for epoch in range(17,19):
                model.train()
                train_loss = 0.0
                
                for inputs, targets in train_dataloader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                
                train_loss /= len(train_dataset)
                
                model.eval()
                valid_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in valid_dataloader:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        valid_loss += loss.item() * inputs.size(0)
                
                valid_loss /= len(valid_dataset)
                
                print(f"Epoch {epoch+1}/{30} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f}")
                
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= 3:
                        break
            
            # テストデータの評価
            test_dataset = TensorDataset(torch.Tensor(input_test.transpose(0,3,1,2)), torch.Tensor(ans_test))
            test_dataloader = DataLoader(test_dataset, batch_size=None, shuffle=False)
            
            model.eval()
            test_loss = 0.0
            predicts = []
            
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item() * inputs.size(0)
                    predicts.append(outputs.numpy())
            
            test_loss /= len(test_dataset)
            predicts = np.concatenate(predicts)
            
            print('predict =', predicts.shape)
            
            # 標準化を元に戻す
            predicts = predicts * ans_std + ans_mean
            
            # モデルデータの保存
            torch.save(model.state_dict(), output_dir + f'/model/model_test{seed:03}.pt')
            
            # 評価データの保存
            np.savez(output_dir + f'geosciAI24/predict/predict_test{seed:03}.npz', 
                    predict=predicts, ans=ans_test, score=test_loss)

        
