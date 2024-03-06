"""
plot the value dist
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
from torch.utils.data import DataLoader
from matplotlib.ticker import FuncFormatter
import sys
sys.path.append('..')
import func.process as pro
import func.net as net
import func.draw as draw


def ax_dist(iss, l, c, ax, bins, alpha):
    ax.bar(x=np.arange(bins), height=iss, color=c, edgecolor="black", linewidth=2, alpha=alpha, label=l)
    return ax


def dist_two(xs, ls, cs, yns, bins, jump, v_name, t, fig_si, fo_si, fo_ti_si, fo_l_si, x_name, alpha):
    fig = plt.figure(figsize=fig_si)
    x1, x2 = xs[0], xs[1]
    l1, l2 = ls[0], ls[1]
    c1, c2 = cs[0], cs[1]
    yn1, yn2 = yns[0], yns[1]
    mid_all_sort, iss1, _, _ = draw.cal_dist(x1, bins, np.min(x1), np.max(x1))
    mid_all_sort, iss2, _, _ = draw.cal_dist(x2, bins, np.min(x2), np.max(x2))

    ax1 = fig.add_subplot(111)
    ax1 = ax_dist(iss1, l1, c1, ax1, bins, alpha)
    ax1.set_ylabel(yn1, fontsize=fo_ti_si, labelpad=20)
    ax1.tick_params(axis='y', labelsize=fo_ti_si)

    ax2 = ax1.twinx()
    ax2 = ax_dist(iss2, l2, c2, ax2, bins, alpha)
    ax2.set_ylabel(yn2, fontsize=fo_ti_si, labelpad=30)
    ax2.tick_params(axis='y', labelsize=fo_ti_si)

    ax1.set_title(t, fontsize=fo_si, pad=30)
    ax1.patch.set_facecolor('grey')
    ax1.patch.set_alpha(0.3)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(c="white", linewidth=5)
    ax1.xaxis.grid(c="white", linewidth=5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=fo_l_si)

    labels = mid_all_sort[np.arange(0, bins, jump)]
    if v_name in ["valid_length"]:
        labels = labels.astype(int)
    ticks = np.arange(bins)[np.arange(0, bins, jump)]
    ax1.set_xticks(ticks=ticks, labels=labels)
    ax1.tick_params(axis='x', labelsize=fo_ti_si, direction='in')
    ax1.set_xlabel(x_name, fontsize=fo_ti_si, labelpad=20)
    fig.subplots_adjust(bottom=0.15, left=0.2, right=0.8)
    return fig


def get_value(df, v_name):
    if v_name in ["source_magnitude"]:
        fig_name = "sm"
        v = df[v_name].values.reshape(-1)
    elif v_name == "valid_length":
        fig_name = "vl"
        v_p_at = df['p_arrival_sample'].values.reshape(-1)
        v_coda_at = pro.read_coda(df)
        v = v_coda_at - v_p_at
    else:
        raise TypeError("Unknown type of 'v_name'!")
    return v, fig_name


train_ratio = 0.75
m = 200000
sm_scale = ["md"]
random = False
v_name = "valid_length"
fig_si = (16, 12)
fo_si = 45
fo_ti_si = 45
fo_te_si = 45
fo_l_si = 45
bins = 45
jump = 9
g_ad = "../graph/magnitude"
la = "zh"
save_fig = True

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
data_train, sm_train, df_train, sm_scale_name = pro.remain_sm_scale(data_train, df_train, sm_train, sm_scale)
data_test, sm_test, df_test, _ = pro.remain_sm_scale(data_test, df_test, sm_test, sm_scale)

v_train, fig_name = get_value(df_train, v_name)
v_test, _ = get_value(df_test, v_name)

# modify title and value
if sm_scale_name == "ml":
    t = "$m_{L}$"
    v_train = np.where(v_train > 0, v_train, 0)
    v_test = np.where(v_test > 0, v_test, 0)
elif sm_scale_name == "md":
    t = "$m_{D}$"
else:
    raise TypeError("!")

# modify x_label, y_label, x_name
if la == "en":
    ls = ['Training set', 'Test set']
    yns = ['Frequency of training set/k', 'Frequency of test set/k']
    if v_name == "source_magnitude":
        x_name = "Magnitude"
    elif v_name == "valid_length":
        x_name = "Valid Length"
    else:
        raise TypeError("Unknown type of 'v_name'")
elif la == "zh":
    ls = ['训练集', '测试集']
    yns = ['训练集样本频次', '测试集样本频次']
    if v_name == "source_magnitude":
        x_name = "震级值"
    elif v_name == "valid_length":
        x_name = "有效分量时段长度"
    else:
        raise TypeError("Unknown type of 'v_name'")
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
else:
    raise TypeError("Unknown type of 'la'")

fig_two = dist_two([v_train, v_test], ls, ['red', 'blue'], yns, bins, jump,
                   v_name, t, fig_si, fo_si, fo_ti_si, fo_l_si, x_name, 0.4)

if save_fig:
    fig_two.savefig(osp.join(g_ad, "dist_{}_{}_{}_{}.png".format(fig_name, sm_scale_name, name, m)))
    pass

print()
plt.show()
print()
