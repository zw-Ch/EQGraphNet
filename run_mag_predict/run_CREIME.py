"""
Magnitude Prediction
[CREIME—A Convolutional Recurrent Model for Earthquake Identification and Magnitude Estimation]
"""
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
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
from func.net import CREIME


def cal_mag(output):
    output_last = output[:, -10:]
    mag = torch.mean(output_last, dim=1)
    return mag


def get_xy(data, df, sm, p_len):
    data, sm = data.numpy(), sm.numpy()
    num = data.shape[0]
    p_as = df["p_arrival_sample"].values.reshape(-1).astype(int)
    n_len = 512 - p_len
    y_n_i = np.ones(shape=(1, n_len)) * (-4)
    x, y = np.zeros(shape=(num, 3, 512)), np.zeros(shape=(num, 512))
    for i in range(num):
        p_as_i, sm_i = p_as[i], sm[i]
        if p_as_i > n_len:
            x_i = data[i, :, (p_as_i - n_len): (p_as_i + p_len)]
            y_i = np.hstack([y_n_i, np.ones(shape=(1, p_len)) * sm_i])
        else:
            x_i = data[i, :, :512]
            y_i = np.hstack([np.ones(shape=(1, p_as_i)) * (-4), np.ones(shape=(1, 512 - p_as_i)) * sm_i])

        x[i, :, :] = x_i
        y[i, :] = y_i
    x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    return x, y


device = "cuda:1" if torch.cuda.is_available() else "cpu"
lr = 0.0005
weight_decay = 0.0005
batch_size = 512
epochs = 100
train_ratio = 0.75
m = 200000                           # number of samples
sm_scale = "ml"                     # magnitude scale
save_txt = True
save_model = True
save_np = True
save_fig = True
random = False
fig_si = (12, 12)          # The size of figures
p_len = 125
fo_si = 40
fo_ti_si = 30
bins = 40
jump = 8

re_ad = osp.join("../result/mag_predict", "CREIME")
if not(osp.exists(re_ad)):
    os.makedirs(re_ad)

"""
Selection of noise and earthquake signals
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
data_train, sm_train, df_train = pro.remain_sm_scale(data_train, df_train, sm_train, sm_scale)
data_test, sm_test, df_test = pro.remain_sm_scale(data_test, df_test, sm_test, sm_scale)

x_train, y_train = get_xy(data_train, df_train, sm_train, p_len)
x_test, y_test = get_xy(data_test, df_test, sm_test, p_len)

pos_train = df_train.loc[:, ["source_longitude", "source_latitude"]].values
pos_test = df_test.loc[:, ["source_longitude", "source_latitude"]].values
trace_train = df_train['trace_name'].values.reshape(-1)
trace_test = df_test['trace_name'].values.reshape(-1)

train_dataset = pro.SelfData(x_train, y_train, sm_train, pos_train, trace_train)
test_dataset = pro.SelfData(x_test, y_test, sm_test, pos_test, trace_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = CREIME().to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

train_pred, train_true, test_pred, test_true = [], [], [], []
train_trace, test_trace = [], []
train_pos, test_pos = [], []
for epoch in range(epochs):
    loss_train_all, loss_test_all = 0, 0
    for item_train, (x_train, y_train, sm_train, pos_train, trace_train, _) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)
        sm_train = sm_train.to(device)

        optimizer.zero_grad()
        output_train = model(x_train)
        loss_train = criterion(output_train, y_train)
        loss_train.backward()
        optimizer.step()
        loss_train_all = loss_train_all + loss_train.item()

        train_pred_one = cal_mag(output_train).detach().cpu().numpy()
        train_true_one = sm_train.detach().cpu().numpy()
        train_trace_one = np.array([trace_train]).reshape(-1, 1)
        train_pos_one = pos_train.numpy()
        if item_train == 0:
            train_pred = train_pred_one
            train_true = train_true_one
            train_trace = train_trace_one
            train_pos = train_pos_one
        else:
            train_pred = np.concatenate((train_pred, train_pred_one), axis=0)
            train_true = np.concatenate((train_true, train_true_one), axis=0)
            train_trace = np.concatenate((train_trace, train_trace_one), axis=0)
            train_pos = np.concatenate((train_pos, train_pos_one), axis=0)

    for item_test, (x_test, y_test, sm_test, pos_test, trace_test, _) in enumerate(test_loader):
        x_test, y_test = x_test.to(device), y_test.to(device)
        sm_test = sm_test.to(device)

        output_test = model(x_test)
        loss_test = criterion(output_test, y_test)
        loss_test_all = loss_test_all + loss_test.item()

        test_pred_one = cal_mag(output_test).detach().cpu().numpy()
        test_true_one = sm_test.detach().cpu().numpy()
        test_trace_one = np.array([trace_test]).reshape(-1, 1)
        test_pos_one = pos_test.numpy()
        if item_test == 0:
            test_pred = test_pred_one
            test_true = test_true_one
            test_trace = test_trace_one
            test_pos = test_pos_one
        else:
            test_pred = np.concatenate((test_pred, test_pred_one), axis=0)
            test_true = np.concatenate((test_true, test_true_one), axis=0)
            test_trace = np.concatenate((test_trace, test_trace_one), axis=0)
            test_pos = np.concatenate((test_pos, test_pos_one), axis=0)

    rmse_train = net.cal_rmse_one_arr(train_true, train_pred)
    rmse_test = net.cal_rmse_one_arr(test_true, test_pred)
    r2_train = net.cal_r2_one_arr(train_true, train_pred)
    r2_test = net.cal_r2_one_arr(test_true, test_pred)
    # r2_train = r2_score(train_true, train_pred)
    # r2_test = r2_score(test_true, test_pred)
    print("Epoch: {:04d}  RMSE_Train: {:.4f}  RMSE_Test: {:.4f}  R2_Train: {:.8f}  R2_Test: {:.8f}".
          format(epoch, rmse_train, rmse_test, r2_train, r2_test))

if save_np:
    np.save(osp.join(re_ad, "train_true_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), train_true)
    np.save(osp.join(re_ad, "train_pred_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), train_pred)
    np.save(osp.join(re_ad, "train_trace_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), train_trace)
    np.save(osp.join(re_ad, "train_pos_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), train_pos)
    np.save(osp.join(re_ad, "test_true_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), test_true)
    np.save(osp.join(re_ad, "test_pred_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), test_pred)
    np.save(osp.join(re_ad, "test_trace_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), test_trace)
    np.save(osp.join(re_ad, "test_pos_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), test_pos)
if save_model:
    torch.save(model.state_dict(), osp.join(re_ad, "model_{}_{}_{}_{}.pkl".format(sm_scale, name, m_train, m_test)))

if save_txt:
    info_txt_address = osp.join(re_ad, "CREIME_result.txt")
    info_df_address = osp.join(re_ad, "CREIME_result.csv")
    f = open(info_txt_address, 'a')
    if osp.getsize(info_txt_address) == 0:
        f.write("r2_test r2_train rmse_test rmse_train p_len sm_scale batch_size epochs lr m_train m_test name\n")
    f.write(str(round(r2_test, 4)) + "  ")
    f.write(str(round(r2_train, 4)) + "  ")
    f.write(str(round(rmse_test, 4)) + "  ")
    f.write(str(round(rmse_train, 4)) + "  ")
    f.write(str(p_len) + "  ")
    f.write(str(sm_scale) + "  ")
    f.write(str(batch_size) + "  ")
    f.write(str(epochs) + "  ")
    f.write(str(lr) + "  ")
    f.write(str(m_train) + "  ")
    f.write(str(m_test) + "  ")
    f.write(str(name) + "  ")

    f.write("\n")
    f.close()

    info = np.loadtxt(info_txt_address, dtype=str)
    columns = info[0, :].tolist()
    values = info[1:, :]
    info_df = pd.DataFrame(values, columns=columns)
    info_df.to_csv(info_df_address)

"""
plot errors and results
"""
fig_train_result = draw.result_fast(train_true, train_pred, True, fig_si, fo_si, fo_ti_si)
fig_test_result = draw.result_fast(test_true, test_pred, False, fig_si, fo_si, fo_ti_si)

train_error = train_pred - train_true
fig_train_error = draw.dist_fast(train_error, bins, jump, "$e^{train}$", fig_si, fo_si, fo_ti_si)
test_error = test_pred - test_true
fig_test_error = draw.dist_fast(test_error, bins, jump, "$e^{test}$", fig_si, fo_si, fo_ti_si)

if save_fig:
    fig_train_result.savefig(osp.join(re_ad, "train_result_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    fig_test_result.savefig(osp.join(re_ad, "test_result_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    fig_train_error.savefig(osp.join(re_ad, "train_error_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    fig_test_error.savefig(osp.join(re_ad, "test_error_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))

print()
plt.show()
print()
