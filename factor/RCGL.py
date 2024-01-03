"""
Studying the impact of RCGL
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
from torch.utils.data import DataLoader
import sys
sys.path.append('..')
import func.process as pro
import func.net as net
import func.draw as draw


class RCGL(nn.Module):
    def __init__(self, gnn_style, adm_style, k, device, num):
        super(RCGL, self).__init__()
        self.relu = nn.ReLU()
        self.gnn_style = gnn_style
        self.adm_style = adm_style
        self.k, self.device = k, device
        self.num = num
        self.pre = nn.Sequential(nn.ReLU())
        self.cnn1 = nn.Conv1d(3, 16, kernel_size=2, stride=2)
        self.cnn2 = nn.Conv1d(16, 16, kernel_size=2, stride=2)
        self.cnn3 = nn.Conv1d(16, 16, kernel_size=2, stride=2)
        self.cnn4 = nn.Conv1d(16, 32, kernel_size=2, stride=2)
        self.cnn5 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn6 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn7 = nn.Conv1d(32, 64, kernel_size=2, stride=2)
        self.cnn8 = nn.Conv1d(64, 64, kernel_size=2, stride=2)
        self.cnn9 = nn.Conv1d(64, 64, kernel_size=2, stride=2)
        self.cnn10 = nn.Conv1d(64, 128, kernel_size=2, stride=2)
        self.cnn11 = nn.Conv1d(128, 128, kernel_size=2, stride=2)
        self.linear = nn.Linear(256, 1)
        self.ei1, self.ew1 = net.get_edge_info(k, 3000, adm_style, device)
        self.ei2, self.ew2 = net.get_edge_info(k, 1500, adm_style, device)
        self.ei3, self.ew3 = net.get_edge_info(k, 750, adm_style, device)
        self.ei4, self.ew4 = net.get_edge_info(k, 375, adm_style, device)
        self.ei5, self.ew5 = net.get_edge_info(k, 187, adm_style, device)
        self.ei6, self.ew6 = net.get_edge_info(k, 93, adm_style, device)
        self.ei7, self.ew7 = net.get_edge_info(k, 46, adm_style, device)
        self.ei8, self.ew8 = net.get_edge_info(k, 23, adm_style, device)
        self.ei9, self.ew9 = net.get_edge_info(k, 11, adm_style, device)
        self.ei10, self.ew10 = net.get_edge_info(k, 5, adm_style, device)
        self.gnn1 = net.get_gnn(gnn_style, 16, 16)
        self.gnn2 = net.get_gnn(gnn_style, 16, 16)
        self.gnn3 = net.get_gnn(gnn_style, 16, 16)
        self.gnn4 = net.get_gnn(gnn_style, 32, 32)
        self.gnn5 = net.get_gnn(gnn_style, 32, 32)
        self.gnn6 = net.get_gnn(gnn_style, 32, 32)
        self.gnn7 = net.get_gnn(gnn_style, 64, 64)
        self.gnn8 = net.get_gnn(gnn_style, 64, 64)
        self.gnn9 = net.get_gnn(gnn_style, 64, 64)
        self.gnn10 = net.get_gnn(gnn_style, 128, 128)

    def forward(self, x):
        h_0 = h = self.cnn1(x)
        if self.num >= 1:
            h = net.run_gnn(self.gnn_style, self.gnn1, h, self.ei1, self.ew1)
            h = h + h_0
        h_1 = h = self.cnn2(self.pre(h))
        if self.num >= 2:
            h = net.run_gnn(self.gnn_style, self.gnn2, h, self.ei2, self.ew2)
            h = h + h_1
        h_2 = h = self.cnn3(self.pre(h))
        if self.num >= 3:
            h = net.run_gnn(self.gnn_style, self.gnn3, h, self.ei3, self.ew3)
            h = h + h_2
        h_3 = h = self.cnn4(self.pre(h))
        if self.num >= 4:
            h = net.run_gnn(self.gnn_style, self.gnn4, h, self.ei4, self.ew4)
            h = h + h_3
        h_4 = h = self.cnn5(self.pre(h))
        if self.num >= 5:
            h = net.run_gnn(self.gnn_style, self.gnn5, h, self.ei5, self.ew5)
            h = h + h_4
        h_5 = h = self.cnn6(self.pre(h))
        if self.num >= 6:
            h = net.run_gnn(self.gnn_style, self.gnn6, h, self.ei6, self.ew6)
            h = h + h_5
        h_6 = h = self.cnn7(self.pre(h))
        if self.num >= 7:
            h = net.run_gnn(self.gnn_style, self.gnn7, h, self.ei7, self.ew7)
            h = h + h_6
        h_7 = h = self.cnn8(self.pre(h))
        if self.num >= 8:
            h = net.run_gnn(self.gnn_style, self.gnn8, h, self.ei8, self.ew8)
            h = h + h_7
        h_8 = h = self.cnn9(self.pre(h))
        if self.num >= 9:
            h = net.run_gnn(self.gnn_style, self.gnn9, h, self.ei9, self.ew9)
            h = h + h_8
        h_9 = h = self.cnn10(self.pre(h))
        if self.num >= 10:
            h = net.run_gnn(self.gnn_style, self.gnn10, h, self.ei10, self.ew10)
            h = h + h_9
        h = self.cnn11(self.pre(h))

        out = h.view(h.shape[0], -1)
        out = self.linear(out)
        return out.view(-1)


device = "cuda:1" if torch.cuda.is_available() else "cpu"
epochs = 70
train_ratio = 0.75
m = 200000                           # number of samples
sm_scale = "md"
lr = 0.0005
save_txt = True
fig_si = (12, 12)          # The size of figures
fo_si = 40
fo_ti_si = 30
random = False

num = 0

save_ad = osp.join("../factor/RCGL_result")
if not(osp.exists(save_ad)):
    os.makedirs(save_ad)

"""
Data Preparation
"""
m_train = int(m * train_ratio)       # number of training samples
m_test = m - m_train                     # number of testing samples
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
data_train, sm_train, df_train, _ = pro.remain_sm_scale(data_train, df_train, sm_train, sm_scale)
data_test, sm_test, df_test, _ = pro.remain_sm_scale(data_test, df_test, sm_test, sm_scale)

train_dataset = pro.SelfData(data_train, sm_train)
test_dataset = pro.SelfData(data_test, sm_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

model = RCGL("gcn", "ts_un", 1, device, num).to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_pred, train_true, test_pred, test_true = [], [], [], []
for epoch in range(epochs):
    loss_train_all, loss_test_all = 0, 0
    for item_train, (x_train, y_train, _) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)

        optimizer.zero_grad()
        output_train = model(x_train)
        loss_train = criterion(output_train, y_train)
        loss_train.backward()
        optimizer.step()
        loss_train_all = loss_train_all + loss_train.item()

        train_pred_one = output_train.detach().cpu().numpy()
        train_true_one = y_train.detach().cpu().numpy()
        if item_train == 0:
            train_pred = train_pred_one
            train_true = train_true_one
        else:
            train_pred = np.concatenate((train_pred, train_pred_one), axis=0)
            train_true = np.concatenate((train_true, train_true_one), axis=0)

    for item_test, (x_test, y_test, _) in enumerate(test_loader):
        x_test, y_test = x_test.to(device), y_test.to(device)

        output_test = model(x_test)
        loss_test = criterion(output_test, y_test)
        loss_test_all = loss_test_all + loss_test.item()

        test_pred_one = output_test.detach().cpu().numpy()
        test_true_one = y_test.detach().cpu().numpy()
        if item_test == 0:
            test_pred = test_pred_one
            test_true = test_true_one
        else:
            test_pred = np.concatenate((test_pred, test_pred_one), axis=0)
            test_true = np.concatenate((test_true, test_true_one), axis=0)

    rmse_train = net.cal_rmse_one_arr(train_true, train_pred)
    rmse_test = net.cal_rmse_one_arr(test_true, test_pred)
    r2_train = net.cal_r2_one_arr(train_true, train_pred)
    r2_test = net.cal_r2_one_arr(test_true, test_pred)
    # r2_train = r2_score(train_true, train_pred)
    # r2_test = r2_score(test_true, test_pred)
    print("Epoch: {:04d}  RMSE_Train: {:.4f}  RMSE_Test: {:.4f}  R2_Train: {:.8f}  R2_Test: {:.8f}".
          format(epoch, rmse_train, rmse_test, r2_train, r2_test))

train_error = train_pred - train_true
test_error = test_pred - test_true

e_m_train, e_m_test = np.mean(train_error), np.mean(test_error)
e_std_train, e_std_test = np.std(train_error), np.std(test_error)

if save_txt:
    info_txt_ad = osp.join(save_ad, "RCGL_result_{}_{}.txt".format(name, m))
    info_df_ad = osp.join(save_ad, "RCGL_result_{}_{}.csv".format(name, m))
    f = open(info_txt_ad, 'a')
    if osp.getsize(info_txt_ad) == 0:
        f.write("num r2 rmse e_m e_std sm_scale epoch name m_train m_test\n")
    f.write(str(num) + "  ")
    f.write(str(round(r2_test, 4)) + "  ")
    f.write(str(round(rmse_test, 4)) + "  ")
    f.write(str(round(e_m_test, 4)) + "  ")
    f.write(str(round(e_std_test, 4)) + "  ")
    f.write(str(sm_scale) + "  ")
    f.write(str(epochs) + "  ")
    f.write(str(name) + "  ")
    f.write(str(m_train) + "  ")
    f.write(str(m_test) + "  ")

    f.write("\n")
    f.close()

