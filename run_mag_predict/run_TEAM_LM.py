"""
Magnitude Prediction
[Earthquake operation and location estimation from real time seismic waveforms with a transformer network]
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import sys
sys.path.append('..')
import func.process as pro
import func.net as net
import func.draw as draw


device = "cuda:0" if torch.cuda.is_available() else "cpu"
lr = 0.0005
weight_decay = 0.0005
batch_size = 64
epochs = 100
train_ratio = 0.75                  # ratio of training samples
m = 40000                           # number of samples
sm_scale = "ml"                     # operation scale
save_txt = False
save_model = True
save_np = True
save_fig = True
random = False
fig_si = (12, 12)          # The size of figures
fo_si = 40
fo_ti_si = 30
bins = 40
jump = 8

re_ad = osp.join("../result/mag_predict", "TEAM_LM")
if not(osp.exists(re_ad)):
    os.makedirs(re_ad)

"""
Construct samples
"""
m_train = int(m * train_ratio)                      # number of training samples
m_test = m - m_train                                # number of testing samples
name = "chunk2"
root = "/home/chenziwei2021/standford_dataset/{}".format(name)

if not random:
    np.random.seed(100)
idx_train, idx_test = pro.get_train_or_test_idx(m, m_train)
eq_train = pro.Chunk(m, True, m_train, idx_train, root, name)
eq_test = pro.Chunk(m, False, m_train, idx_test, root, name)
df_train, df_test = eq_train.df, eq_test.df

data_train, data_test = eq_train.data.float(), eq_test.data.float()
sm_train = torch.from_numpy(df_train["source_magnitude"].values.reshape(-1)).float()
sm_test = torch.from_numpy(df_test["source_magnitude"].values.reshape(-1)).float()

# Select samples according to Magnitude Type
data_train, sm_train, df_train = pro.remain_sm_scale(data_train, df_train, sm_train, sm_scale)
data_test, sm_test, df_test = pro.remain_sm_scale(data_test, df_test, sm_test, sm_scale)

# The location of sources
df_train_pos = df_train.loc[:, ["source_longitude", "source_latitude"]].values
df_test_pos = df_test.loc[:, ["source_longitude", "source_latitude"]].values

train_dataset = pro.SelfData(data_train, sm_train)
test_dataset = pro.SelfData(data_test, sm_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

"""
Construct model for training and testing
"""
model = net.TEAMLM().to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

train_pred, train_true, test_pred, test_true = [], [], [], []
train_pos, test_pos = [], []
for epoch in range(epochs):
    loss_train_all, loss_test_all = 0, 0
    for item_train, (x_train, y_train, index_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)

        optimizer.zero_grad()
        output_train = model(x_train)
        loss_train = criterion(output_train, y_train)
        loss_train.backward()
        optimizer.step()
        loss_train_all = loss_train_all + loss_train.item()

        train_pred_one = output_train.detach().cpu().numpy()
        train_true_one = y_train.detach().cpu().numpy()
        train_index_one = index_train.numpy()
        train_pos_one = df_train_pos[train_index_one, :]
        if item_train == 0:
            train_pred = train_pred_one
            train_true = train_true_one
            train_pos = train_pos_one
        else:
            train_pred = np.concatenate((train_pred, train_pred_one), axis=0)
            train_true = np.concatenate((train_true, train_true_one), axis=0)
            train_pos = np.concatenate((train_pos, train_pos_one), axis=0)

    for item_test, (x_test, y_test, index_test) in enumerate(test_loader):
        x_test, y_test = x_test.to(device), y_test.to(device)

        output_test = model(x_test)
        loss_test = criterion(output_test, y_test)
        loss_test_all = loss_test_all + loss_test.item()

        test_pred_one = output_test.detach().cpu().numpy()
        test_true_one = y_test.detach().cpu().numpy()
        test_index_one = index_test.numpy()
        test_pos_one = df_test_pos[test_index_one, :]
        if item_test == 0:
            test_pred = test_pred_one
            test_true = test_true_one
            test_pos = test_pos_one
        else:
            test_pred = np.concatenate((test_pred, test_pred_one), axis=0)
            test_true = np.concatenate((test_true, test_true_one), axis=0)
            test_pos = np.concatenate((test_pos, test_pos_one), axis=0)

    rmse_train = net.cal_rmse_one_arr(train_true, train_pred)
    rmse_test = net.cal_rmse_one_arr(test_true, test_pred)
    r2_train = net.cal_r2_one_arr(train_true, train_pred)
    r2_test = net.cal_r2_one_arr(test_true, test_pred)
    # r2_train = r2_score(train_true, train_pred)
    # r2_test = r2_score(test_true, test_pred)
    print("Epoch: {:04d}  RMSE_Train: {:.4f}  RMSE_Test: {:.4f}  R2_Train: {:.8f}  R2_Test: {:.8f}".
          format(epoch, rmse_train, rmse_test, r2_train, r2_test))



print()
plt.show()
print()
