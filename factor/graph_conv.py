"""
Analyzing the Properties of Graph Convolution
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from pygsp import graphs
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

import sys
sys.path.append('..')
import func.process as pro
import func.net as net


def find_min_snr(df, data):
    snr = df['snr_db'].values.reshape(-1)
    snr_max, index, j = -float('inf'), 0, 0
    for i in range(len(snr)):
        snr_one = snr[i][1:-1].split(' ')
        num_full = snr_one.count('')
        idx = 0
        while idx < num_full:  # 除去空格
            snr_one.remove('')
            idx = idx + 1
        snr_one = np.array(snr_one).astype(float)

        if np.max(snr_one) > snr_max:
            index = i
            j = np.argmax(snr_one)
            snr_max = np.max(snr_one)
    return data[index, j, :].numpy()


def get_u(ad, a):
    u_ad = osp.join(ad, "u.npy")
    if not osp.exists(u_ad):
        g = graphs.Graph(a)
        u = g.U
        np.save(u_ad, u)
    else:
        u = np.load(u_ad)
    return u.transpose()

def graph_conv(x, a, num):
    d = np.diag(np.sum(a, axis=1))      # degree matrix
    d_norm = np.power(d, 0.5)
    x = x.reshape(-1, 1)
    for _ in tqdm(range(num)):
        x = np.dot(np.dot(np.dot(d_norm, a), d_norm), x)
    return x.reshape(-1)


def get_noise_one(x, ratio):
    e = np.mean(np.square(x))
    e_n = e / ratio
    n = np.random.normal(loc=0, scale=np.sqrt(e_n), size=x.shape)
    n = np.expand_dims(n, axis=0)
    return n


def gft(u, x):
    x = x.reshape(-1, 1)
    x_gft = np.dot(u, x)
    return x_gft.reshape(-1)


def thousands(x, pos):
    return '%1.0f' % (x * 1e-4)


def compare(x, x_n, num, y_lim, fig_si, fo_si):
    alpha = 0.5
    formatter = FuncFormatter(thousands)
    fig, ax = plt.subplots(1, 1, figsize=fig_si)
    if num == 0:
        word = "无图卷积，"
    else:
        word = "{}次图卷积，".format(num)
    ax.plot(gft(u, graph_conv(x, a, num)), label=word + "地震信号", alpha=alpha)
    ax.plot(gft(u, graph_conv(x_n, a, num)), label=word + "加噪信号", alpha=alpha)
    # ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.patch.set_facecolor('grey')
    ax.patch.set_alpha(0.3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(c="white", linewidth=2)
    ax.xaxis.grid(c="white", linewidth=2)
    plt.legend(fontsize=fo_si)
    if num == 20:
        plt.xlabel("图频率", fontsize=fo_si, labelpad=10)
    plt.xticks(fontsize=fo_si)
    plt.ylabel("图傅里叶变换/10$^{4}$", fontsize=fo_si, labelpad=1)
    plt.yticks(fontsize=fo_si)
    plt.locator_params(axis='y', nbins=5)
    plt.ylim(y_lim)
    fig.subplots_adjust(left=0.2, bottom=0.2)
    return fig


train_ratio = 0.75
item_x = 14
item_n = 10
idx = 1
fig_si = (20, 4)
fo_si = 20
la = "zh"
random = False
save_fig = True
g_ad = "../graph/eq_features"
re_ad = "../factor/graph_conv"
if not(osp.exists(re_ad)):
    os.makedirs(re_ad)

"""
Data Preparation
"""
if not random:
    np.random.seed(0)
m = 100                                                       # number of samples
m_train = int(m * train_ratio)
name_no, name_eq = "chunk1", "chunk2"
root_eq = "/home/chenziwei2021/standford_dataset/{}".format(name_eq)
# root_no = "/home/chenziwei2021/standford_dataset/{}".format(name_no)

idx_train, idx_test = pro.get_train_or_test_idx(m, m_train)
eq = pro.Chunk(m, True, m_train, idx_train, root_eq, name_eq)
# no = pro.Chunk(m, True, m_train, idx_train, root_no, name_no)
eq_data, df = eq.data, eq.df

"""
plot gft
"""
if la == "zh":
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

x = find_min_snr(df, eq_data)       # earthquake signal
n = get_noise_one(x, 10)            # noise
x_n = x + n                         # noisy signal
length = x.shape[0]              # length of earthquake signal
a = net.ts_un(length, 1)            # adjacency matrix
u = get_u(re_ad, a)           # gft matrix

y_lim = (-100 * 1e3, 125 * 1e3)

fig_ori = compare(x, x_n, 0, y_lim, fig_si, fo_si)
fig_less = compare(x, x_n, 5, y_lim, fig_si, fo_si)
fig_more = compare(x, x_n, 20, y_lim, fig_si, fo_si)

if save_fig:
    fig_ori.savefig(osp.join(g_ad, "gc_ori.png"))
    fig_less.savefig(osp.join(g_ad, "gc_less.png"))
    fig_more.savefig(osp.join(g_ad, "gc_more.png"))
    pass

print()
plt.show()
print()
