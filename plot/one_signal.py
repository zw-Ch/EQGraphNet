"""
Plot for one signal (noise)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import os.path as osp
import sys
sys.path.append("..")
import func.process as pro
import func.draw as draw


def ax_one_x(ax, x_, fo_ti, fo_ti_si, ti_style, p_at, s_at, c_at, la):
    ax.plot(x_, c="black")
    ax.set_ylim([-np.max(np.abs(x_) * 1.1), np.max(np.abs(x_)) * 1.1])
    ax.tick_params(axis=ti_style, labelsize=fo_ti_si, direction='in', length=5, width=1.5)
    ymin, ymax = ax.get_ylim()
    if la == "en":
        ax.vlines(p_at, ymin * 0.8, ymax * 0.8, color='b', lw=4, label='P arrival')
        ax.vlines(s_at, ymin * 0.8, ymax * 0.8, color='r', lw=4, label='S arrival')
        ax.vlines(c_at, ymin * 0.8, ymax * 0.8, color='green', lw=4, label='Coda end')
    elif la == "zh":
        ax.vlines(p_at, ymin * 0.8, ymax * 0.8, color='b', lw=4, label='P波到时')
        ax.vlines(s_at, ymin * 0.8, ymax * 0.8, color='r', lw=4, label='S波到时')
        ax.vlines(c_at, ymin * 0.8, ymax * 0.8, color='green', lw=4, label='尾波结束时')
    ax.legend(fontsize=fo_ti, loc=1)
    return ax


def one_x(x, x_df, fig_si, fo_si, fo_ti_si, la):
    fig, axes = plt.subplots(3, 1, figsize=fig_si)
    p_at, s_at, c_at = int(x_df.p_arrival_sample), int(x_df.s_arrival_sample), int(x_df.coda_end_sample[2:-3])
    x_e, x_n, x_z = x[0, :], x[1, :], x[2, :]
    axes[0] = ax_one_x(axes[0], x_e, fo_si, fo_ti_si, 'both', p_at, s_at, c_at, la)
    axes[1] = ax_one_x(axes[1], x_n, fo_si, fo_ti_si, 'both', p_at, s_at, c_at, la)
    axes[2] = ax_one_x(axes[2], x_z, fo_si, fo_ti_si, 'both', p_at, s_at, c_at, la)
    plt.setp(axes[0].get_xticklabels(), visible=False)
    plt.setp(axes[1].get_xticklabels(), visible=False)
    if la == "en":
        fig.supylabel("Amplitude counts", fontsize=fo_si, x=0.01)
    elif la == "zh":
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig.supylabel("地震信号振幅", fontsize=fo_si, x=0.01)
    # fig.subplots_adjust(hspace=0.1)
    fig.tight_layout()
    return fig


def ax_one_n(ax, n_, fo_si, fo_ti_si, ti_style):
    ax.plot(n_, c="black")
    ax.set_ylim([-np.max(np.abs(n_) * 1.1), np.max(np.abs(n_)) * 1.1])
    ax.tick_params(axis=ti_style, labelsize=fo_si, direction='in', length=5, width=1.5)
    return ax


def one_n(n, fig_si, fo_si, fo_ti_si):
    fig, axes = plt.subplots(3, 1, figsize=fig_si)
    n_e, n_n, n_z = n[0, :], n[1, :], n[2, :]
    axes[0] = ax_one_n(axes[0], n_e, fo_si, fo_ti_si, 'both')
    axes[1] = ax_one_n(axes[1], n_n, fo_si, fo_ti_si, 'both')
    axes[2] = ax_one_n(axes[2], n_z, fo_si, fo_ti_si, 'both')
    plt.setp(axes[0].get_xticklabels(), visible=False)
    plt.setp(axes[1].get_xticklabels(), visible=False)
    # fig.supylabel("Amplitude counts", fontsize=fo_si)
    fig.subplots_adjust(hspace=0.1)
    return fig


def ax_one_in(ax, x_):
    ax.plot(x_, c="black")
    ax.axis('off')
    return ax


def one_in(x, fig_si):
    fig, axes = plt.subplots(3, 1, figsize=fig_si)
    x_e, x_n, x_z = x[0, :], x[1, :], x[2, :]
    axes[0] = ax_one_in(axes[0], x_e)
    axes[1] = ax_one_in(axes[1], x_n)
    axes[2] = ax_one_in(axes[2], x_z)
    fig.subplots_adjust(hspace=0.1)
    return fig


train_ratio = 0.75
item_x = 14
item_n = 10
fig_si = (17, 11)          # The size of figures
fo_si = 35
fo_ti_si = 35
random = False
g_ad = "../graph/eq_features"
la = "zh"
save_fig = True

"""
Data Preparation
"""
if not random:
    np.random.seed(0)
m_no, m_eq = 100, 100                                                       # number of samples
m_no_train, m_eq_train = int(m_no * train_ratio), int(m_eq * train_ratio)       # number of training samples
m_no_test, m_eq_test = m_no - m_no_train, m_eq - m_eq_train                     # number of testing samples
name_no, name_eq = "chunk1", "chunk2"
root_no = "/home/chenziwei2021/standford_dataset/{}".format(name_no)
root_eq = "/home/chenziwei2021/standford_dataset/{}".format(name_eq)

idx_train_no, idx_test_no = pro.get_train_or_test_idx(m_no, m_no_train)
no_train = pro.Chunk(m_no, True, m_no_train, idx_train_no, root_no, name_no)
no_test = pro.Chunk(m_no, False, m_no_train, idx_test_no, root_no, name_no)
no_train_data, no_test_data = no_train.data, no_test.data

idx_train_eq, idx_test_eq = pro.get_train_or_test_idx(m_eq, m_eq_train)
eq_train = pro.Chunk(m_eq, True, m_eq_train, idx_train_eq, root_eq, name_eq)
eq_test = pro.Chunk(m_eq, False, m_eq_train, idx_test_eq, root_eq, name_eq)
eq_train_data, eq_test_data = eq_train.data, eq_test.data

x, n = eq_train_data[item_x, :, :].numpy(), no_train_data[item_n, :, :].numpy()
x_df = eq_train.df.iloc[item_x, :]

"""
plot one sample
"""
fig_x = one_x(x, x_df, fig_si, fo_si, fo_ti_si, la)
# fig_n = one_n(n, fig_si, fo_si, fo_ti_si)

"""
plot one sample as input
"""
# fig_x_in = one_in(x, (16, 4))
# fig_n_in = one_in(n, (6, 4))

if save_fig:
    pass
    fig_x.savefig(osp.join(g_ad, "example_earth.png"))
    # fig_n.savefig(osp.join(g_ad, "example_noise.png"))
    # fig_x_in.savefig(osp.join(g_ad, "input_x.png"))
    # fig_n_in.savefig(osp.join(g_ad, "input_n.png"))

print()
plt.show()
print()
