"""
Analyse the effect of earthquake characteristics in earthquake information
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
from matplotlib.ticker import FuncFormatter
import sys
sys.path.append('..')
import func.output as out
import func.draw as draw


def dist(fea, bins, jump, fig_si, fo_si, fo_ti_si, x_name, y_name, la, v_min=None, v_max=None,
         alpha=1, c='royalblue'):
    fig = plt.figure(figsize=fig_si)
    ax = fig.add_subplot(111)
    mid_all_sort, interval_sum_sort, left, right = draw.cal_dist(fea, bins, v_min, v_max)
    ax.bar(x=np.arange(bins), height=interval_sum_sort, color=c, edgecolor="black", linewidth=2, alpha=alpha)

    def thousands(x, pos):
        return '%1.0f' % (x * 1e-3)

    formatter = FuncFormatter(thousands)
    ax.yaxis.set_major_formatter(formatter)

    labels = mid_all_sort[np.arange(0, bins, jump)]
    ticks = np.arange(bins)[np.arange(0, bins, jump)]
    # plt.title(t, fontsize=fo_si, pad=30)
    plt.xticks(ticks=ticks, labels=labels, fontsize=fo_ti_si)
    plt.yticks(fontsize=fo_ti_si)
    if la == "zh":
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
    if x_name is not None:
        plt.xlabel(x_name, fontsize=fo_si, labelpad=30)
    if y_name is not None:
        plt.ylabel(y_name, fontsize=fo_si, labelpad=20)
    fig.subplots_adjust(bottom=0.2)
    return fig


def tran_fea(fea_):
    fea = fea_.astype(float)
    return fea


def read_snr(df, style):
    snr = df.snr_db.values
    num = snr.shape[0]
    snr_all = []
    for i in range(num):
        snr_one = snr[i][1:-1]
        snr_one_ = snr_one.split(' ')
        num_full = snr_one_.count('')
        idx = 0
        while idx < num_full:  # 除去空格
            snr_one_.remove('')
            idx = idx + 1
        snr_one_ = np.array([float(snr_one_[0]), float(snr_one_[1]), float(snr_one_[2])])
        if style == "mean":  # 以平均值作为标签label
            snr_one_mean = np.mean(snr_one_)
            snr_all.append(snr_one_mean)
        else:
            raise TypeError("Unknown type of style")
    snr_all = np.array(snr_all)
    return snr_all


# "snr_db", "source_distance_km", "source_depth_km"
fea_name = "source_depth_km"

# "EQGraphNet", "MagInfoNet"
model = "EQGraphNet"

fea_range = [""]
sm_list = ["ml", "md"]
name = "chunk2"
root = "/home/chenziwei2021/standford_dataset/{}".format(name)
m = 200000
re_ad = "../result/mag_predict"
train_ratio = 0.75
m_train = int(m * train_ratio)
m_test = m - m_train
fo_si = 50
fo_ti_si = 45
bins = 40
jump = 8
save_fig = True
la = "zh"
g_ad = "../graph/eq_features"

cmax = 0.19
y_lim = (-10, 2)
x_name, y_name, fig_name, v_max = "", "", "", 0

if not(osp.exists(g_ad)):
    os.makedirs(g_ad)

"""
Data preparation
"""
pos, true, trace, pred_MaI, pred_EQG, _, _, _ = out.read_sm(
    sm_list, re_ad, name, m_train, m_test)

df = pd.read_csv(osp.join(root, name + ".csv"))
df = out.select_trace(trace, df, re_ad, name, m)

if fea_name == "snr_db":
    fea = read_snr(df, style="mean")
else:
    fea = df[fea_name].values.reshape(-1)
fea = tran_fea(fea)

if fea_name == "source_distance_km" and la == "zh":
    x_name = "震中距（千米）"
elif fea_name == "source_depth_km" and la == "zh":
    x_name = "震源深度（千米）"
elif fea_name == "snr_db" and la == "zh":
    x_name = "信噪比（dB）"
elif fea_name == "source_distance_km" and la == "en":
    x_name = "Hypocentral distances (km)"
elif fea_name == "source_depth_km" and la == "en":
    x_name = "Hypocentral depths (km)"
elif fea_name == "snr_db" and la == "en":
    x_name = "SNR (dB)"

if la == "zh":
    y_name = "频次/千"
elif la == "en":
    y_name = "Frequency"

if fea_name == "source_distance_km":
    fig_name = "source_distance"
    v_max = 0.061
elif fea_name == "source_depth_km":
    fig_name = "source_depth"
    v_max = 0.192
elif fea_name == "snr_db":
    fig_name = "snr"
    v_max = 0.121

"""
errors on different features
"""
if model == "EQGraphNet":
    error = pred_EQG - true
elif model == "MagInfoNet":
    error = pred_MaI - true
else:
    raise TypeError("Unknown type of model! Must be 'EQGraphNet' or 'MagInfoNet'!")

# fig_cb = draw.color_bar(np.array([1, 2]), np.array([1, 2]), "核密度", (60, 12), fo_si, la, cmax)
fig_error_fea = draw.error_fea(error, fea, (20, 36), fo_si, fo_ti_si, cmax, x_name, True, y_lim, la, model, v_max)
# fig_fea_dist = dist(fea, 40, 8, (20, 7), fo_si, fo_ti_si, x_name, y_name, la)

if save_fig:
    pass
    # fig_cb.savefig(osp.join(g_ad, "cb.png"))
    fig_error_fea.savefig(osp.join(g_ad, model, "error_fea_{}.png".format(fig_name)))
    # fig_fea_dist.savefig(osp.join(g_ad, "dist_{}.png".format(fig_name)))

print()
plt.show()
print()
