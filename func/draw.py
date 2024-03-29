"""
Functions for plot and draw figures
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
os.environ['PROJ_LIB'] = '<path to anaconda>/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde


def cal_rmse_one_arr(true, pred):
    return np.sqrt(np.mean(np.square(true - pred)))


def cal_r2_one_arr(true, pred):
    corr_matrix = np.corrcoef(true, pred)
    corr = corr_matrix[0, 1]
    r2 = corr ** 2
    return r2


def cal_dist(x, bins, v_min=None, v_max=None):
    if v_min is not None:
        x = x[x >= v_min]
        x = np.append(x, v_min)
    if v_max is not None:
        x = x[x <= v_max]
        x = np.append(x, v_max)
    x_label, x_bins = pd.cut(x, bins=bins, retbins=True)
    x_label_vc = pd.DataFrame(x_label).value_counts()
    interval = x_label_vc.index.tolist()
    interval_sum = x_label_vc.values
    mid_all, left, right = [], float('inf'), -float('inf')
    for i in range(bins):
        interval_one = interval[i][0]
        left_one, right_one = interval_one.left, interval_one.right
        mid = (left_one + right_one) / 2
        mid_all.append(mid)
        if left_one < left:
            left = left_one
        if right_one > right:
            right = right_one
    mid_all = np.array(mid_all)
    sort_index = np.argsort(mid_all)
    mid_all_sort = mid_all[sort_index]
    mid_all_sort = np.around(mid_all_sort, 2)
    interval_sum_sort = interval_sum[(sort_index)]
    if v_min is not None:
        interval_sum_sort[0] = interval_sum_sort[0] - 1
    if v_max is not None:
        interval_sum_sort[-1] = interval_sum_sort[-1] - 1
    return mid_all_sort, interval_sum_sort, left, right


def dist_fast(x, bins, jump, x_name, fig_si, fo_si, fo_ti_si, t=None, y_name="Frequency"):
    mid_all_sort, interval_sum_sort, left, right = cal_dist(x, bins)         # calculate dist
    fig = plt.figure(figsize=fig_si)
    if t is not None:
        plt.title(t, fontsize=fo_si)
    plt.bar(x=np.arange(bins), height=interval_sum_sort)
    labels = mid_all_sort[np.arange(0, bins, jump)]
    ticks = np.arange(bins)[np.arange(0, bins, jump)]
    plt.xticks(ticks=ticks, labels=labels, fontsize=fo_ti_si)
    plt.yticks(fontsize=fo_ti_si)
    if x_name is not None:
        plt.xlabel(x_name, fontsize=fo_si)
    if y_name is not None:
        plt.ylabel(y_name, fontsize=fo_si)
    return fig


def result_fast(true, predict, train, fig_si, fo_si, fo_ti_si):
    fig = plt.figure(figsize=fig_si)
    plt.scatter(true, predict, s=5)
    line = r2_line(true, predict, None, None)
    line_min, line_max = np.min(line), np.max(line)
    plt.plot(line, line, c="black")
    if train:
        plt.xlabel("$y^{train}$", fontsize=fo_si)
        plt.ylabel("$\hat{y}^{train}$", fontsize=fo_si)
    else:
        plt.xlabel("$y^{test}$", fontsize=fo_si)
        plt.ylabel("$\hat{y}^{test}$", fontsize=fo_si)
    plt.xticks(fontsize=fo_ti_si)
    plt.yticks(fontsize=fo_ti_si)
    plt.xlim(line_min - 0.2, line_max + 0.2)
    plt.ylim(line_min - 0.2, line_max + 0.2)
    return fig


def r2_line(y1, y2, mag_min, mag_max):
    y1_min, y1_max = np.min(y1), np.max(y1)
    y2_min, y2_max = np.min(y2), np.max(y2)
    y_min = np.max(np.array([y1_min, y2_min]))
    y_max = np.min(np.array([y1_max, y2_max]))
    y1_len, y2_len = y1.shape[0], y2.shape[0]
    y_len = np.max(np.array([y1_len, y2_len]))
    if mag_min is not None:
        y_min = mag_min
    if mag_max is not None:
        y_max = mag_max
    line = np.linspace(y_min, y_max, y_len)
    return line


def result(true, pred, loc, sm, fig_si, fo_si, fo_ti_si, fo_te, la, den=False, cmax=None, mag_min=None, mag_max=None,
           c='lightcoral', t=None):
    line = r2_line(true, pred, mag_min, mag_max)
    line_min, line_max = np.min(line), np.max(line)
    fig = plt.figure(figsize=fig_si)
    ax = fig.add_subplot(111)
    lim_min, lim_max = line_min - 0.2, line_max + 0.2
    _, cs = ax_result(ax, line, sm, true, pred, loc, fo_te, la, c, den, cmax)

    if la == "zh":
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.xlabel("真实震级", fontsize=fo_si, labelpad=20)
        plt.ylabel("估计震级", fontsize=fo_si, labelpad=20)
    elif la == "en":
        plt.xlabel("True Magnitudes", fontsize=fo_si, labelpad=20)
        plt.ylabel("Estimated Magnitudes", fontsize=fo_si, labelpad=20)
    if t is not None:
        plt.title(t, fontsize=fo_si, pad=30)
    plt.xticks(fontsize=fo_ti_si)
    plt.yticks(fontsize=fo_ti_si)
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    # cbar = fig.colorbar(cs, ax=ax, location='top', pad=0.03, shrink=0.8)
    # cbar.set_label("Kernel Density", fontsize=fo_si, labelpad=20)
    # cbar.ax.tick_params(labelsize=fo_si - 10)
    fig.subplots_adjust(bottom=0.15)
    return fig


def ax_result(ax, line, sm, true, pred, loc, fo_te, la, c, den, cmax):
    r2 = cal_r2_one_arr(true, pred)
    rmse = cal_rmse_one_arr(true, pred)
    if la == "zh":
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        z_label = "零误差线"
        f_label = "线性回归"
        c_label = " 样本"
    elif la == "en":
        z_label = "Zero error line"
        f_label = "Regression line"
        c_label = " samples"
    else:
        raise TypeError("!")
    ax.plot(line, line, ls=(0, (5, 1)), c="black", lw=8, label=z_label)

    # fit line
    fit_x = np.linspace(np.min(line), np.max(line), 100)
    linear = LinearRegression()
    linear.fit(true.reshape(-1, 1), pred.reshape(-1, 1))
    fit_y = linear.predict(fit_x.reshape(-1, 1)).reshape(-1)
    ax.plot(fit_x, fit_y, ls="solid", c="saddlebrown", lw=8, label=f_label)

    if den:
        pos = np.vstack([true.reshape(1, -1), pred.reshape(1, -1)])
        pos_c = gaussian_kde(pos)(pos)
        cs = ax.scatter(true, pred, c=pos_c, alpha=0.7, s=300, cmap=plt.cm.rainbow, label=sm + c_label, vmax=cmax)
    else:
        cs = ax.scatter(true, pred, c=c, alpha=0.7, s=300, label=sm + c_label, vmax=cmax)
    ax.patch.set_facecolor('grey')
    ax.patch.set_alpha(0.3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(c="white", linewidth=5)
    ax.xaxis.grid(c="white", linewidth=5)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.legend(fontsize=fo_te - 15, loc=4)
    if loc != []:
        if la == "zh":
            t_str = "R$^{2}$" + "   = {:.4f}\nRMSE = {:.4f}".format(r2, rmse)
        elif la == "en":
            t_str = "R$^{2}$" + "      = {:.4f}\nRMSE = {:.4f}".format(r2, rmse)
        else:
            raise ValueError("!")
        t = ax.text(loc[0], loc[1], t_str, ha='center', va='center', fontsize=fo_te, transform=ax.transAxes)
        t.set_bbox(dict(facecolor=c, alpha=0.5, edgecolor=c))
    return ax, cs


def color_bar(true, pred, t, fig_si, fo_si, la, cmax=None):
    fig = plt.figure(figsize=fig_si)
    ax = fig.add_subplot(111)
    c_list = [0.01, cmax]
    cs = ax.scatter(true, pred, c=c_list, alpha=0.7, s=300, cmap=plt.cm.rainbow, label=t + " samples")
    if la == "zh":
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
    cbar = fig.colorbar(cs, ax=ax, location='bottom', pad=0.03, shrink=0.8, aspect=100)
    cbar.set_label(t, fontsize=fo_si, labelpad=20)
    cbar.ax.tick_params(labelsize=fo_si - 10)
    fig.subplots_adjust(bottom=0.15)
    return fig


def dist(x, bins, jump, pos, t, fig_si, fo_si, fo_ti_si, fo_te_si, la, x_name, y_name, v_min=None, v_max=None,
         alpha=1, c='royalblue'):
    fig = plt.figure(figsize=fig_si)
    ax = fig.add_subplot(111)

    def thousands(x, pos):
        if t == "$m_{L}$":
            return '%1.0f' % (x * 1e-3)
        elif t == "$m_{D}$":
            return '%1.1f' % (x * 1e-3)
        else:
            raise TypeError("!")

    formatter = FuncFormatter(thousands)
    ax.yaxis.set_major_formatter(formatter)
    ax.locator_params(nbins=6, axis='y')

    mid_all_sort, interval_sum_sort, left, right = cal_dist(x, bins, v_min, v_max)
    ax_dist(ax, interval_sum_sort, x, bins, pos, fo_te_si, c, alpha)
    labels = mid_all_sort[np.arange(0, bins, jump)]
    ticks = np.arange(bins)[np.arange(0, bins, jump)]
    # plt.title(t, fontsize=fo_si, pad=30)
    plt.xticks(ticks=ticks, labels=labels, fontsize=fo_ti_si)
    plt.yticks(fontsize=fo_ti_si)
    if la == "zh":
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
    plt.xlabel(x_name, fontsize=fo_si, labelpad=30)
    plt.ylabel(y_name, fontsize=fo_si, labelpad=30)
    fig.subplots_adjust(bottom=0.2, left=0.15)
    return fig


def ax_dist(ax, interval_sum_sort, x, bins, pos, fo_te_si, c, alpha):
    ax.bar(x=np.arange(bins), height=interval_sum_sort, color=c, edgecolor="black", linewidth=2, alpha=alpha)
    ax.patch.set_facecolor('grey')
    ax.patch.set_alpha(0.3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(c="white", linewidth=5)
    ax.xaxis.grid(c="white", linewidth=5)
    if pos != []:
        mean, std = np.mean(x), np.std(x)
        t = ax.text(pos[0], pos[1], r'$\mu^{ e}$' + " = {:.4f}\n".format(mean) + r'$\sigma^{ e}$' + " = {:.4f}".format(std),
                    ha='center', va='center', transform=ax.transAxes, fontsize=fo_te_si)
        t.set_bbox(dict(facecolor='lightcoral', alpha=0.5, edgecolor='lightcoral'))
    return ax


def map_view(arr_list, t_list, pos, lat_min, lat_max, lon_min, lon_max, fig_si, fo_si, fo_ti_si, vmin=None, vmax=None):
    num = len(arr_list)
    num_row, num_col = 2, 3         # display style
    if num != num_row * num_col:
        raise ValueError("The length of arr_list must be equal to num_row * num_col! Please modify it")
    fig = plt.figure(figsize=fig_si)
    axes = fig.subplots(num_row, num_col)
    for i in range(num):
        ax = axes[i // num_col, i % num_col]
        arr = arr_list[i].tolist()
        t = t_list[i]
        ax.set_title(t, fontsize=fo_si, pad=10)
        m = Basemap(ax=ax,
                    llcrnrlon=lon_min,
                    llcrnrlat=lat_min,
                    urcrnrlon=lon_max,
                    urcrnrlat=lat_max)
        m.arcgisimage(service='World_Imagery', xpixels=800, verbose=True)    # may be unavailable due to net
        x, y = m(pos[:, 0], pos[:, 1])
        # m.drawcoastlines()
        # m.drawmapboundary()
        parallels = np.linspace(lat_min, lat_max, 4)[1:-1]
        meridians = np.linspace(lon_min, lon_max, 6)[1:-1]

        # only show: latitude for left and longitude for bottom
        if i % num_col == 0:
            m.drawparallels(np.around(parallels, 0), labels=[1, 0, 0, 0], fontsize=fo_ti_si)
        if i // num_col == (num_row - 1):
            m.drawmeridians(np.around(meridians, 0), labels=[0, 0, 0, 1], fontsize=fo_ti_si)

        cs = m.scatter(x, y, marker='o', c=arr, s=80, linewidth=2, cmap=plt.cm.turbo, vmin=vmin, vmax=vmax)
    fig.subplots_adjust(wspace=0.2)
    cbar = fig.colorbar(cs, ax=axes, location='right', pad=0.02, shrink=0.8)
    cbar.set_label("Magnitudes", fontsize=fo_si, labelpad=20)
    cbar.ax.tick_params(labelsize=fo_ti_si)
    return fig


# based on seismic parameter (snr, depth, distance...) plot estimated errors
def error_fea(error, fea, fig_si, fo_si, fo_ti_si, cmax, x_name, mean, y_lim, la, model, vmax):
    fig = plt.figure(figsize=fig_si)
    ax = fig.add_subplot(111)
    pos = np.hstack([error.reshape(-1, 1), fea.reshape(-1, 1)])
    pos_c = gaussian_kde(pos.T)(pos.T)
    abc = np.max(pos_c)     # EQG and MaI use same 'vmax', defined in feature_effect.py
    cs = ax.scatter(fea, error, c=pos_c, s=300, alpha=0.7, cmap=plt.cm.rainbow, vmax=vmax)
    if mean:
        error_mean, fea_mean = error_fea_mean(error, fea)
        ax.scatter(fea_mean, error_mean, c="brown", s=200, alpha=0.9, marker="s")
    ax.hlines(0, xmin=np.min(fea), xmax=np.max(fea), colors="black", ls='dashed', lw=6)
    ax.patch.set_facecolor('grey')
    ax.patch.set_alpha(0.3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(c="white", linewidth=5)
    ax.xaxis.grid(c="white", linewidth=5)
    cbar = fig.colorbar(cs, ax=ax, location='top', pad=0.03, shrink=0.8, aspect=75)
    if la == "zh":
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        cbar.set_label("核密度", fontsize=fo_si - 5, labelpad=20)
        ax.set_ylabel("{}估计误差".format(model), fontsize=fo_si, labelpad=20)
    elif la == "en":
        cbar.set_label("Kernel Density", fontsize=fo_si - 5, labelpad=20)
        ax.set_ylabel("{} estimated errors".format(model), fontsize=fo_si, labelpad=20)
    else:
        raise TypeError("!")
    cbar.ax.tick_params(labelsize=fo_ti_si - 5)
    if x_name is not None:
        ax.set_xlabel(x_name, fontsize=fo_si, labelpad=30)
    # ax.xaxis.set_label_position("top")
    # ax.xaxis.tick_top()

    ax.set_yticks(np.arange(y_lim[0], y_lim[1] + 1))
    ax.tick_params(axis='both', labelsize=fo_ti_si, length=1)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    # plt.subplots_adjust(bottom=0.15)
    return fig


def error_fea_mean(error_, fea_, bins=60):
    idx_sort = np.argsort(fea_)
    error, fea = error_[idx_sort].tolist(), fea_[idx_sort].tolist()
    fea_min, fea_max = np.min(fea_), np.max(fea_)
    fea_bins = np.linspace(fea_min, fea_max, bins)
    error_mean, fea_mean = [], []
    for i in range(bins - 1):
        left, right = fea_bins[i], fea_bins[i + 1]
        e_in = []
        while len(error) != 0:
            error_j, fea_j = error[0], fea[0]
            if (fea_j >= left) & (fea_j <= right):
                e_in.append(error_j)
                del error[0]
                del fea[0]
            else:
                break
        e_in = np.array(e_in)
        if e_in.shape[0] == 0:
            continue
        else:
            error_mean.append(np.mean(e_in))
            fea_mean.append((left + right) / 2)
    error_mean, fea_mean = np.array(error_mean), np.array(fea_mean)
    return error_mean, fea_mean


def loss(l_MaI_train, l_MaI_test, l_EQG_train, l_EQG_test, sm, fig_si, fo_si, fo_ti_si, fo_l_si, la, v_lim=None):
    fig, ax = plt.subplots(1, 1, figsize=fig_si)
    if la == "en":
        label_train = "Training"
        label_test = "Testing"
        x_label = "Epoch"
        y_label = "Loss"
    elif la == "zh":
        label_train = "训练集"
        label_test = "测试集"
        x_label = "迭代轮数"
        y_label = "均方损失"
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
    else:
        raise TypeError("Unknown type of 'la', must be 'zh' or 'en'!")
    alpha, lw = 0.9, 4
    ax.plot(l_MaI_train, lw=lw, label="MagInfoNet" + label_train, c='red', alpha=alpha)
    ax.plot(l_MaI_test, lw=lw, label="MagInfoNet" + label_test, c='lightcoral', alpha=alpha)
    ax.plot(l_EQG_train, lw=lw, label="EQGraphNet" + label_train, c='purple', alpha=alpha)
    ax.plot(l_EQG_test, lw=lw, label="EQGraphNet" + label_test, c='orchid', alpha=alpha)
    ax.set_xlabel(x_label, fontsize=fo_si, labelpad=30)
    ax.set_ylabel(y_label, fontsize=fo_si, labelpad=30)
    ax.legend(fontsize=fo_l_si)
    x_ticks = np.array([9, 19, 29, 39, 49, 59, 69])
    ax.set_xticks(x_ticks - 2)
    ax.set_xticklabels(x_ticks + 1)
    ax.set_title(sm, fontsize=fo_si, pad=30)
    if v_lim is not None:
        ax.set_ylim(v_lim[0], v_lim[1])
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', labelsize=fo_ti_si)
    fig.subplots_adjust(bottom=0.18, left=0.15)
    return fig


# replace value in array (beautifully plot)
def rep_arr(ratio, arr, loc, value):
    if len(loc) != len(value):
        raise ValueError("length of loc and value are not same!")
    if not isinstance(loc, list) or not isinstance(value, list):
        raise TypeError("loc and value must be 'list'!")
    for loc_one, value_one in zip(loc, value):
        arr[loc_one] = value_one * ratio
    return arr
