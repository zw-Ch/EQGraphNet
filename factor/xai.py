"""
Explainable Artificial Intelligence
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import sys
sys.path.append('..')
import func.process as pro
import func.net as net
import func.draw as draw
from func.net import EQGraphNet


def tran_ew(*args):
    num = len(args)
    outs = []
    for i in range(num):
        arg = args[i]
        out = arg.detach().cpu().numpy()
        outs.append(out)
    return tuple(outs)


def edge(ei, ew):
    ew, ei = ew.detach().cpu().numpy(), ei.detach().cpu().numpy()
    num_nodes = np.unique(ei[0, :]).shape[0]
    ed = []
    for i in range(num_nodes):
        idx = np.argwhere(ei[0, :] == i).reshape(-1)
        # a_ws = np.mean(np.abs(e_w[idx]))
        ed_i = np.mean(ew[idx])
        ed.append(ed_i)
    ed = np.array(ed)
    return ed


device = "cuda:1" if torch.cuda.is_available() else "cpu"
batch_size = 64
adm_style = "ts_un"
gnn_style = "gcn"
k = 1
train_ratio = 0.75
m = 200000                           # number of samples
sm_scale = "ml"                     # magnitude scale
random = False
re_ad = "../result/mag_predict/EQGraphNet"
save_ad = "xai_result"

m_train = int(m * train_ratio)       # number of training samples
m_test = m - m_train                     # number of testing samples
name = "chunk2"
root = "/home/chenziwei2021/standford_dataset/{}".format(name)

"""
Data Preparation
"""
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

# pos_train = df_train.loc[:, ["source_longitude", "source_latitude"]].values
# pos_test = df_test.loc[:, ["source_longitude", "source_latitude"]].values
# trace_train = df_train['trace_name'].values.reshape(-1)
# trace_test = df_test['trace_name'].values.reshape(-1)
#
# train_dataset = pro.SelfData(data_train, sm_train, pos_train, trace_train)
# test_dataset = pro.SelfData(data_test, sm_test, pos_test, trace_test)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

"""
Layers
"""
# model = EQGraphNet(gnn_style, adm_style, k, device).to(device)
# model.load_state_dict(torch.load(osp.join(re_ad, "model_{}_{}_{}_{}.pkl".format(sm_scale, name, m_train, m_test))))
#
# c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = model.cnn1, model.cnn2, model.cnn3, model.cnn4, model.cnn5, model.cnn6, \
#                                                model.cnn7, model.cnn8, model.cnn9, model.cnn10, model.cnn11
# g1, g2, g3, g4, g5, g6, g7, g8, g9, g10 = model.gnn1, model.gnn2, model.gnn3, model.gnn4, model.gnn5, model.gnn6, \
#                                           model.gnn7, model.gnn8, model.gnn9, model.gnn10
# ei1, ei2, ei3, ei4, ei5, ei6, ei7, ei8, ei9, ei10 = model.ei1, model.ei2, model.ei3, model.ei4, model.ei5, model.ei6, \
#                                                     model.ei7, model.ei8, model.ei9, model.ei10
# ew1, ew2, ew3, ew4, ew5, ew6, ew7, ew8, ew9, ew10 = model.ew1, model.ew2, model.ew3, model.ew4, model.ew5, model.ew6, \
#                                                     model.ew7, model.ew8, model.ew9, model.ew10
# linear, relu = model.linear, model.relu
#
# ew1 = edge(ei1, ew1)
#
# plt.figure()
# plt.plot(ew1)

"""
Decomposition
"""
s_ad = osp.join(save_ad, "x_{}.npy".format(sm_scale))
if osp.exists(s_ad):
    s_dc = np.load(s_ad)
else:
    s_train, s_test = data_train.numpy(), data_test.numpy()
    s = np.concatenate((s_train, s_test), axis=0)
    s = s.reshape(s.shape[0], -1)
    tsne = TSNE(n_components=2)
    s_dc = tsne.fit_transform(s)
    np.save(s_ad, s_dc)

plt.figure()
plt.scatter(s_dc[:, 0], s_dc[:, 1], s=3, alpha=0.5)


print()
plt.show()
print()
