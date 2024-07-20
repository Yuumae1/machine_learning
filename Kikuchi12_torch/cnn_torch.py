import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optimizers
from sklearn.utils import shuffle
#import torchvision
#from torchvision import transforms
#from torch.utils.data import DataLoader
#from PIL import Image
import EarlyStopping

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def normalization(data):
  data_mean = np.mean(data, axis=0)
  data_std  = np.std(data, axis=0)
  data_norm = (data - data_mean) / data_std
  data_norm = np.nan_to_num(data_norm, nan=0) # 欠損値(nan)を0で置換
  del data_mean, data_std
  return data_norm

def indexing(lead_time):
  output_shape = 2
  rt = real_time2[:-lead_time-1]
  t_data = PC_norm[lead_time:]
  print(t_data.shape)
  idx = np.where((rt.year <= 2015))[0]
  t_train = t_data[idx]
  idx = np.where((rt.year > 2015))[0]
  t_test = t_data[idx]
  print('t_train, t_test', t_train.shape, t_test.shape)
  return data, rt, t_train, t_test, output_shape

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


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutions(input_channels, output_channels, kernel_size, stride, padding)
        self.c1 = nn.Conv2d(8*3, 32, kernel_size=3, stride=2, padding=1)
        self.b1 = nn.BatchNorm2d(32)
        self.a1 = nn.ReLU()
        self.d1 = nn.Dropout(0.2)
        self.c2 = nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=1)
        self.b2 = nn.BatchNorm2d(64)
        self.a2 = nn.ReLU()
        self.d2 = nn.Dropout(0.2)
        self.c3 = nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1)
        self.b3 = nn.BatchNorm2d(128)
        self.a3 = nn.ReLU()
        self.d3 = nn.Dropout(0.2)
        self.fc = nn.Linear(128*4*4, 2)
        

    def forward(self, x):
        x = self.d1(self.a1(self.b1(self.c1(x))))
        x = self.d2(self.a2(self.b2(self.c2(x))))
        x = self.d3(self.a3(self.b3(self.c3(x))))
        x = x.view(-1, 128*4*4)
        x = self.fc(x)
        return x


def culc_cor(predict, y_test, lead_time):
  cor = (np.sum(predict[:,0] * y_test[:,0], axis=0) + np.sum(predict[:,1] * y_test[:,1], axis=0)) / \
          (np.sqrt(np.sum(predict[:,0] ** 2 + predict[:,1] ** 2, axis=0)) * np.sqrt(np.sum(y_test[:,0] ** 2 + y_test[:,1] ** 2, axis=0)))
  print('lead time {} day = '.format(lead_time), cor)

def learning_curve(history, lead_time):
  plt.figure(figsize=(8, 6))
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')    #Validation loss : 精度検証データにおける損失
  plt.xlim(0, 200)
  plt.ylim(0, 1.)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Loss vs. Epoch   Lead Time = ' + str(lead_time) + 'days')
  plt.legend()
  plt.savefig(f'/home/maeda/machine_learning/results/kikuchi-8vals_v1/learning_curve/8vals/{(lead_time):03}day.png')
  plt.close()



if __name__ == '__main__':
  np.random.seed(123)
  torch.manual_seed(123)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  #mode = 'mjo'
  mode = 'bsiso'
  data = np.load('/home/maeda/data/bsiso_eeof/prepro_anomaly_8vals.npz')

  olr = data['olr'][80:,24:49,:]
  u850 = data['u850'][80:,24:49,:]
  v850 = data['v850'][80:,24:49,:]
  u200 = data['u200'][80:,24:49,:]
  v200 = data['v200'][80:,24:49,:]
  h850 = data['h850'][80:,24:49,:]
  pr_wtr = data['pr_wtr'][80:,24:49,:]
  sst = data['sst'][80:,24:49,:]

  lat = data['lat'][24:49]
  lon = data['lon']
  time = data['time'][80:]    # 射影後にデータが10日進むため、時刻の方を前進させておく
  real_time = pd.to_datetime(time, unit='h', origin=pd.Timestamp('1800-01-01')) # 時刻をdatetime型に変換


  x = np.stack([olr, u850, v850, u200, v200, h850, pr_wtr, sst], 3)
  x_train = []
  x_test = []
  x = np.array(normalization(x[:,:,:,i]) for i in range(x.shape[3]))

  
  # bsiso index (eEOF) 読み込み
  if mode == 'bsiso':
    data_file = '/home/maeda/data/bsiso_eeof/bsiso_rt-PCs.npz'
  elif mode == 'mjo':
    data_file = '/home/maeda/data/bsiso_eeof/mjo_rt-PCs.npz'
  PC      = np.load(data_file)['rt_PCs'][:,:2]
  sign    = np.array([-1, 1]).T
  PC_norm = sign * PC / PC.std(axis=0)[np.newaxis,:]
  time2   = np.load(data_file)['time']
  real_time2 = pd.to_datetime(time2, unit='h', origin=pd.Timestamp('1800-01-01')) # 時刻をdatetime型に変換
  print('PCs = ', PC_norm.shape)
  print('time PCs= ', time2.shape)
  print('real time PCs = ', real_time2[0], real_time2[-1])
  

  
  '''
  training description
  '''
  
  def train_step(x, t):
      model.train()
      preds = model(x)
      loss = loss_fn(preds, t)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      return loss, preds
    
  def test_step(x, t):
    model.eval()
    preds = model(x)
    loss = loss_fn(preds, t)
    return loss, preds
  

  #lt_box = [0, 5, 10, 15, 20, 25, 30, 35]
  lt_box = np.arange(1)
  
  for lead_time in lt_box:
    print('==== lead time : {} day ====='.format(lead_time))
    
    for i in range(8):
      _x_train, _x_test = preprocess(x[:,:,:,i], real_time, 0)
      x_train.append(_x_train)
      x_test.append(_x_test)
    x_train = np.stack(np.array(x_train), 3)
    x_test = np.stack(np.array(x_test), 3)
    print('x_train, x_test = ', x_train.shape, x_test.shape)
    rt, t_train, t_test, output_shape = indexing(lead_time=0)
    print('rt, t_train, t_test = ', rt.shape, t_train.shape, t_test.shape)

    batch_size = 32
    n_batches = x_train.shape[0] // batch_size
    epoch_num = 200
    
    for seed in range(1):
      print('Seed = ', seed)
      set_seed(seed)
      model = Conv().to(device)
      callback = EarlyStopping(patience=5, 
                               verbose=1, 
                               path=f'/home/maeda/machine_learning/results/model/kikuchi-8vals_v1/8vals/model_{(lead_time):03}day/seed{(seed):03}.hdf5'
                               )
      loss_fn = nn.MSELoss()
      optimizer = optimizers.Adam(model.parameters(), lr=0.001)
      
      for epoch in range(epoch_num):
        train_loss = 0.
        x_, t_ = shuffle(x_train, t_train)
        x_ = torch.Tensor(x_).to(device)
        t_ = torch.Tensor(t_).to(device)
        
        for n_batch in range(n_batches):
            start = n_batch * batch_size
            end = start + batch_size
            loss, preds = train_step(x_[start:end], t_[start:end])
            train_loss += loss.item()
        if epoch % 10 == 0:
            print('epoch: {}, loss: {:.3}'.format(
                epoch+1,
                train_loss
            ))
        if callback.early_stop:
          print('Early stopping')
          break
      

      x_test = torch.Tensor(x_test).to(device)
      t_test = torch.Tensor(t_test).to(device)
      loss, preds_test = test_step(x_test, t_test)
      predict = preds_test.cpu().detach().numpy()
      print(predict.shape)
      print('test loss: {:.3}'.format(loss.item()))
      culc_cor(predict, t_test, lead_time)
      #learning_curve(history, lead_time)
      # model save
      torch.save(model.state_dict(), 'cnn_model.pth')
      if mode == 'bsiso':
        np.savez(f'/home/maeda/machine_learning/results/kikuchi-8vals_v1/cor/8vals/{(lead_time):03}day/torch_seed{(seed):03}.npz', predict, t_test)
        torch.save(model.state_dict(), f'/home/maeda/machine_learning/results/model/kikuchi-8vals_v1/8vals/model_{(lead_time):03}day/seed{(seed):03}.pth')
      elif mode == 'mjo':
        np.savez(f'/home/maeda/machine_learning/results/kikuchi-8vals_mjo/cor/8vals/{(lead_time):03}day/torch_seed{(seed):03}.npz', predict, t_test)
        torch.save(model.state_dict(), f'/home/maeda/machine_learning/results/model/kikuchi-8vals_mjo/8vals/model_{(lead_time):03}day/seed{(seed):03}.pth')

print('==== Finish! ====')