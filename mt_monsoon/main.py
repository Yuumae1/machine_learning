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
from callbacks import EarlyStopping
from torchsummary import summary


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
  print('t_train, t_test =', t_train.shape, t_test.shape)
  return rt, t_train, t_test, output_shape

def preprocess(data, rt, lead_time):
  ipt = data[10:-lead_time-1]
  #ipt_lag5  = data[5:-lead_time-6]
  #ipt_lag10 = data[:-lead_time-11]
  # =========
  # 訓練データの作成(通年データとする)
  idx1 = np.where((rt.year <= 2015))[0]
  ipt_train = ipt[idx1]
  #ipt_lag5_train = ipt_lag5[idx1]
  #ipt_lag10_train = ipt_lag10[idx1]
  #ipt_train = np.stack([ipt_lag0_train, ipt_lag5_train, ipt_lag10_train], 3)

  # 検証データの作成
  idx2 = np.where((rt.year > 2015))[0]
  ipt_test = ipt[idx2]
  #ipt_lag5_test = ipt_lag5[idx2]
  #ipt_lag10_test = ipt_lag10[idx2]
  #ipt_test = ipt[idx]
  #ipt_test = np.concatenate([ipt_lag0_test, ipt_lag5_test, ipt_lag10_test], 1)
  #ipt_test = np.stack([ipt_lag0_test, ipt_lag5_test, ipt_lag10_test], 3)
  return ipt_train, ipt_test


'''
model definition
'''
class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
          nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.Dropout(0.2))
        self.layer2 = nn.Sequential(
          nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=1),
          nn.BatchNorm2d(64), 
          nn.ReLU(),
          nn.Dropout(0.2))
        self.layer3 = nn.Sequential(
          nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Dropout(0.2))
        self.fc1 = nn.Sequential(
          nn.Linear(128*4*19, 128),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Dropout(0.2))
        self.fc2 = nn.Linear(128, 2)
        #self.fc1 = nn.Linear(128*4*19, 2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
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


'''
main program
'''
if __name__ == '__main__':
  np.random.seed(123)
  torch.manual_seed(123)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  #mode = 'mjo'
  #mode = 'bsiso'
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
  x_n = np.zeros(x.shape)
  for i in range(x.shape[3]):
    x_n[:,:,:,i] = normalization(x[:,:,:,i])
  print('x_n = ', x_n.shape)  


  
  # bsiso index (eEOF) 読み込み
  data_wpsh = np.load()
  PC      = np.load(data_wpsh)['rt_PCs'][:,:2]
  sign    = np.array([-1, 1]).T
  PC_norm = sign * PC / PC.std(axis=0)[np.newaxis,:]
  time2   = np.load(data_wpsh)['time']
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
  lt_box = np.arange(0, 31)
  
  for lead_time in lt_box:
    print('==== lead time : {} day ====='.format(lead_time))
    # answer data
    rt, t_train, t_test, output_shape = indexing(lead_time)
    print('rt, t_train, t_test = ', rt.shape, t_train.shape, t_test.shape)
    # input data
    x_train = []
    x_test = []
    for i in range(8):
      _x_train, _x_test = preprocess(x_n[:,:,:,i], rt, lead_time)
      x_train.append(_x_train)
      x_test.append(_x_test)
    
    x_train = np.array(x_train).transpose(1, 0, 2, 3)
    x_test = np.array(x_test).transpose(1, 0, 2, 3)
    print('x_train, x_test =', x_train.shape, x_test.shape)
    rt, t_train, t_test, output_shape = indexing(lead_time)
    print('rt, t_train, t_test =', rt.shape, t_train.shape, t_test.shape)

    batch_size = 128
    n_batches = x_train.shape[0] // batch_size
    epoch_num = 200
    
    for seed in range(10):
      print('Seed = ', seed)
      set_seed(seed)
      model = Conv().to(device)  
      summary(model, (8, 25, 144))
      es = EarlyStopping(patience=5, 
                         verbose=0,   # EalyStopping Counterの表示の有無（0/1）
                         path=f'/home/maeda/machine_learning/results/model/kikuchi-single/8vals/model_{(lead_time):03}day/seed{(seed):03}.pth'
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

        x_test = torch.Tensor(x_test).to(device)
        t_test = torch.Tensor(t_test).to(device)
        test_loss, preds_test = test_step(x_test, t_test)
        print('epoch: {}, loss: {:.3}, test loss: {:.3}'.format(
          epoch+1,
          train_loss/n_batches,
          test_loss.item()
          ))
        
        #if callback.early_stop:
        #  print('Early stopping')
        #  break
        if es(test_loss.item(), model):
          print('Early stopping')
          break
      
      predict = preds_test.cpu().detach().numpy()
      t_test  = t_test.cpu().detach().numpy()
      print(predict.shape)
        
      culc_cor(predict, t_test, lead_time)
      #learning_curve(history, lead_time)
      # model save
      torch.save(model.state_dict(), 'cnn_model.pth')
      if mode == 'bsiso':
        np.savez(f'/home/maeda/machine_learning/results/kikuchi-single/cor/8vals/{(lead_time):03}day/torch_seed{(seed):03}.npz', predict, t_test)
        torch.save(model.state_dict(), f'/home/maeda/machine_learning/results/model/kikuchi-single/8vals/model_{(lead_time):03}day/seed{(seed):03}.pth')
      elif mode == 'mjo':
        np.savez(f'/home/maeda/machine_learning/results/kikuchi-8vals_mjo/cor/8vals/{(lead_time):03}day/torch_seed{(seed):03}.npz', predict, t_test)
        torch.save(model.state_dict(), f'/home/maeda/machine_learning/results/model/kikuchi-8vals_mjo/8vals/model_{(lead_time):03}day/seed{(seed):03}.pth')

print('==== Finish! ====')