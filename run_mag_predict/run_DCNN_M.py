"""
Magnitude Prediction
[Magnitude Estimation for Earthquake Early Warning Using a Deep Convolutional Neural Network]
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


def get_in(x, df):
    x, num = x.numpy(), x.shape[0]
    e, n, z = x[:, 0, :], x[:, 1, :], x[:, 2, :]
    p_as = df["p_arrival_sample"].values.reshape(-1).astype(int)
    p_ts = (df["p_travel_sec"].values.reshape(-1) * 100).astype(int)
    in_all = np.zeros(shape=(num, 12))
    for i in range(num):
        e_i, n_i, z_i = e[i, :], n[i, :], z[i, :]
        p_as_i, p_ts_i = p_as[i], p_ts[i]
        left, right = p_as_i, p_as_i + p_ts_i

        d_ud = z_i
        v_ud = np.diff(d_ud)
        a_ud = np.diff(v_ud)
        a_ew, a_ns = np.diff(np.diff(e_i)), np.diff(np.diff(n_i))

        P_d = np.max(np.abs(d_ud)[left:right])
        P_v = np.max(np.abs(v_ud)[left:right])
        P_a = np.max(np.abs(a_ud)[left:right])

        r = np.sum(np.square(v_ud)[left:right]) / np.sum(np.square(d_ud)[left:right])
        tau_c = 2 * np.pi / np.sqrt(r)
        TP = tau_c * P_d
        T_va = 2 * np.pi * (P_v / P_a)

        PI_v = np.max(np.log(np.abs(a_ud * v_ud[:-1]))[left:right])
        IV2 = np.sum(np.square(v_ud)[left:right])
        a_3 = np.sqrt(np.square(a_ud) + np.square(a_ew) + np.square(a_ns))
        CAV = np.sum(a_3[left:right])

        cvad = np.sum(np.abs(d_ud)[left:right])
        cvav = np.sum(np.abs(v_ud)[left:right])
        cvaa = np.sum(np.abs(a_ud)[left:right])

        in_i = np.array([P_d, P_v, P_a, tau_c, TP, T_va, PI_v, IV2, CAV, cvad, cvav, cvaa]).reshape(1, -1)
        in_all[i, :] = in_i
    return in_all


device = "cuda:1" if torch.cuda.is_available() else "cpu"
lr = 0.0005
weight_decay = 0.0005
batch_size = 64
epochs = 100
hid_dim = 32
adm_style = "ts_un"
gnn_style = "gcn"
k = 1
train_ratio = 0.75
num_nodes = 750
m = 10000                           # number of samples
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

re_ad = osp.join("../result/mag_predict", "DCNN_M")
if not(osp.exists(re_ad)):
    os.makedirs(re_ad)

"""
Selection of noise and earthquake signals
"""
m_train = int(m * train_ratio)       # number of training samples
m_test = m - m_train                     # number of testing samples
name = "chunk2"
root = "/home/chenziwei2021/standford_dataset/{}".format(name)

idx_train, idx_test = pro.get_train_or_test_idx(m, m_train)
eq_train = pro.Chunk(m, True, m_train, idx_train, root, name)
eq_test = pro.Chunk(m, False, m_train, idx_test, root, name)
df_train, df_test = eq_train.df, eq_test.df

data_train, data_test = eq_train.data, eq_test.data
sm_train = torch.from_numpy(df_train["source_magnitude"].values.reshape(-1)).float()
sm_test = torch.from_numpy(df_test["source_magnitude"].values.reshape(-1)).float()
in_train, in_test = get_input(data_train, df_train), get_input(data_test, df_test)

# Select samples according to Magnitude Type
data_train, sm_train, df_train = pro.remain_sm_scale(data_train, df_train, sm_train, sm_scale)
data_test, sm_test, df_test = pro.remain_sm_scale(data_test, df_test, sm_test, sm_scale)

# The location of sources
df_train_pos = df_train.loc[:, ["source_longitude", "source_latitude"]].values
df_test_pos = df_test.loc[:, ["source_longitude", "source_latitude"]].values


print()
plt.show()
print()
