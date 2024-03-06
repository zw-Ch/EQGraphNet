"""
To extract features of different layers from MagInfoNet
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os.path as osp
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import sys
sys.path.append("..")
import func.net as net
import func.process as pro
import func.draw as draw


def plot_scatter(pos, fig_size, x_lim=None, y_lim=None, s=10):
    x, y = pos[:, 0], pos[:, 1]
    fig = plt.figure(figsize=fig_size)
    plt.scatter(x, y, s=s)
    if x_lim is not None:
        plt.xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])
    return fig


def dc(data, re_ad, dc_style, name, n_components):
    data_dc_ad = "{}/dc_{}_{}.npy".format(re_ad, name, dc_style)
    data = data.reshape(data.shape[0], -1)
    if osp.exists(data_dc_ad):          # have already been decomposed
        data_dc = np.load(data_dc_ad)
        return data_dc
    if dc_style == "pca":
        tool = PCA(n_components=n_components)
    elif dc_style == "tsne":
        tool = TSNE(n_components=n_components)
    else:
        raise TypeError("Unknown dc_style")
    data_dc = tool.fit_transform(data)
    np.save(data_dc_ad, data_dc)
    return data_dc


class RM_X(nn.Module):
    def __init__(self, cnn1, cnn2, cnn3, cnn4, pre, bn1, pool1):
        super(RM_X, self).__init__()
        self.cnn1, self.cnn2, self.cnn3, self.cnn4 = cnn1, cnn2, cnn3, cnn4
        self.pre, self.bn1, self.pool1 = pre, bn1, pool1

    def forward(self, x):
        h_x = self.cnn1(x.unsqueeze(1))
        h_x_0 = h_x
        h_x = self.pre(self.bn1(h_x))
        h_x = self.cnn2(h_x)
        h_x = self.pre(self.bn1(h_x))
        h_x = self.cnn3(h_x)
        h_x = self.pre(self.bn1(h_x))
        h_x = h_x + h_x_0
        h_x = self.cnn4(self.bn1(h_x))
        h_x = self.pool1(h_x)
        h_x = h_x.squeeze(1)
        return h_x


class PredInform(nn.Module):
    def __init__(self, l_at, l_t, RM_x, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, p2, pre, bn2):
        super(PredInform, self).__init__()
        self.l_at, self.l_t, self.RM_x = l_at, l_t, RM_x
        self.c5, self.c6, self.c7, self.c8, self.c9, self.c10 = c5, c6, c7, c8, c9, c10
        self.c11, self.c12, self.c13, self.c14, self.c15 = c11, c12, c13, c14, c15
        self.p2, self.pre, self.bn2 = p2, pre, bn2

    def forward(self, x, ps_at, p_t):
        h_at = self.l_at(ps_at).unsqueeze(1)
        h_t = self.l_t(p_t).unsqueeze(1)
        h_x = self.RM_x(x)

        out = torch.cat((h_x, h_at, h_t), dim=1)
        h = out
        out = self.c5(out.unsqueeze(1))
        out_0 = out
        out = self.pre(self.bn2(out))
        out = self.c6(out)
        out = self.pre(self.bn2(out))
        out = self.c7(out)
        out = out + out_0

        out_1 = out
        out = self.c8(self.pre(self.bn2(out)))
        out = self.c9(self.pre(self.bn2(out)))
        out = out + out_1

        out_2 = out
        out = self.c10(self.pre(self.bn2(out)))
        out = self.c11(self.pre(self.bn2(out)))
        out = out + out_2

        out_3 = out
        out = self.c12(self.pre(self.bn2(out)))
        out = self.c13(self.pre(self.bn2(out)))
        out = out + out_3

        out_4 = out
        out = self.c14(self.pre(self.bn2(out)))
        out = self.c15(self.pre(self.bn2(out)))
        out = out + out_4

        out = self.p2(out)
        X_in = out.view(out.shape[0], out.shape[1], -1).permute(0, 2, 1)
        return h, X_in


device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 32
lr = 0.0001                      # learning rate
weight_decay = 5e-4
epochs = 200
train_ratio = 0.75
m = 200000                           # number of samples
sm_scale = "ml"              # operation scale
save_txt = True
save_model = True
save_np = True
save_fig = True
random = False
fig_si = (12, 12)          # The size of figures
fo_si = 40
fo_ti_si = 20
dc_style = "tsne"
prep_style = "sta"
re_ad = "../result/mag_predict/MagInf"

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
data_train, sm_train, df_train, sm_scale_name = pro.remain_sm_scale(data_train, df_train, sm_train, sm_scale)
data_test, sm_test, df_test, _ = pro.remain_sm_scale(data_test, df_test, sm_test, sm_scale)

pos_train = df_train.loc[:, ["source_longitude", "source_latitude"]].values
pos_test = df_test.loc[:, ["source_longitude", "source_latitude"]].values
trace_train = df_train['trace_name'].values.reshape(-1)
trace_test = df_test['trace_name'].values.reshape(-1)

ps_at_name = ["p_arrival_sample", "s_arrival_sample"]
ps_at_train, ps_at_test = df_train.loc[:, ps_at_name].values, df_test.loc[:, ps_at_name].values
prep_ps_at, ps_at_train, ps_at_test = pro.prep_pt(prep_style, ps_at_train, ps_at_test)
ps_at_train, ps_at_test = torch.from_numpy(ps_at_train).float(), torch.from_numpy(ps_at_test).float()

t_name = ["p_travel_sec"]
p_t_train, p_t_test = df_train.loc[:, t_name].values, df_test.loc[:, t_name].values
prep_p_t, p_t_train, p_t_test = pro.prep_pt(prep_style, p_t_train, p_t_test)
p_t_train, p_t_test = torch.from_numpy(p_t_train).float(), torch.from_numpy(p_t_test).float()

"""
Model Layers Preparation
"""
MaI = net.MagInfoNet("unimp", "ts_un", 2, device).to(device)
MaI.load_state_dict(torch.load(osp.join(re_ad, "model_{}_{}_{}_{}.pkl".format(sm_scale, name, m_train, m_test))))

pre, bn1, bn2 = MaI.pre, MaI.bn1, MaI.bn2
c1, c2, c3, c4, pool1 = MaI.cnn1, MaI.cnn2, MaI.cnn3, MaI.cnn4, MaI.pool1
rm_x = RM_X(c1, c2, c3, c4, pre, bn1, pool1).to(device)

c5, c6, c7, c8, c9, c10 = MaI.cnn5, MaI.cnn6, MaI.cnn7, MaI.cnn8, MaI.cnn9, MaI.cnn10
c11, c12, c13, c14, c15, p2 = MaI.cnn11, MaI.cnn12, MaI.cnn13, MaI.cnn14, MaI.cnn15, MaI.pool2

l_at, l_t, last = MaI.linear_at, MaI.linear_t, MaI.last
pred_inform = PredInform(l_at, l_t, rm_x, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, p2, pre, bn2).to(device)

"""
Input Data Dimensionality reduction
"""
test_x = data_test.numpy()
test_x_dc = dc(test_x, re_ad, dc_style, "x", 2)
fig_x = plot_scatter(test_x_dc, fig_si)

"""
The role of RM in the Pred-Inform
"""
test_dataset = pro.SelfData(data_test, sm_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
test_hx = []
test_pos = []
for item_test, (x_test, y_test, index_test) in enumerate(tqdm(test_loader)):
    x_test, y_test = x_test.to(device), y_test.to(device)

    output_test = rm_x(x_test)
    test_hx_one = output_test.detach().cpu().numpy()
    test_index_one = index_test.numpy()
    test_pos_one = pos_test[test_index_one, :]
    if item_test == 0:
        test_hx = test_hx_one
        test_pos = test_pos_one
    else:
        test_hx = np.concatenate((test_hx, test_hx_one), axis=0)
        test_pos = np.concatenate((test_pos, test_pos_one), axis=0)

test_hx_dc = dc(test_hx, re_ad, dc_style, "hx", 2)
fig_hx = plot_scatter(test_hx_dc, fig_si)

"""
The role of RM and UniMP in the Mag-Pred
"""
test_dataset = pro.SelfData(data_test, sm_test, ps_at_test, p_t_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
h, X_in, true = [], [], []
for item_test, (x_test, y_test, ps_at_test, p_t_test, index_test) in enumerate(tqdm(test_loader)):
    x_test, y_test = x_test.to(device), y_test.to(device)
    ps_at_test, p_t_test = ps_at_test.to(device), p_t_test.to(device)

    h_one, X_in_one = pred_inform(x_test, ps_at_test, p_t_test)
    h_one = h_one.detach().cpu().numpy()
    X_in_one = X_in_one.detach().cpu().numpy()
    true_one = y_test.detach().cpu().numpy()
    if item_test == 0:
        h = h_one
        X_in = X_in_one
        true = true_one
    else:
        h = np.concatenate((h, h_one), axis=0)
        X_in = np.concatenate((X_in, X_in_one), axis=0)
        true = np.concatenate((true, true_one), axis=0)

h_dc = dc(h, re_ad, "pca", "h", 600)
X_in_dc = dc(X_in, re_ad, "pca", "X_in", 600)

h_dc = torch.from_numpy(h_dc).to(device)
X_in_dc = torch.from_numpy(X_in_dc).to(device)
last = last.to(device)

predict_h = last(h_dc).detach().cpu().numpy().reshape(-1)
predict_X_in = last(X_in_dc).detach().cpu().numpy().reshape(-1)

r2_h = net.cal_r2_one_arr(true, predict_h)
r2_X_in = net.cal_r2_one_arr(true, predict_X_in)
print("R2_h: {:.8f}\nR2_X_in: {:.8f}".format(r2_h, r2_X_in))

draw.plot_run_result(true, predict_h, False, fig_si, fo_si, fo_ti_si)
draw.plot_run_result(true, predict_X_in, False, fig_si, fo_si, fo_ti_si)

if save_fig:
    fig_x.savefig(osp.join(re_ad, "dc_x_{}.png".format(dc_style)))
    fig_hx.savefig(osp.join(re_ad, "dc_hx_{}.png".format(dc_style)))

print()
plt.show()
print()
