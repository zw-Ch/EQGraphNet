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
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde


def cal_dist(x, bins, v_min=None, v_max=None):
    if v_min is not None:
        x = x[x >= v_min]
        x = np.append(x, v_min)
    if v_max is not None:
        x = x[x <= v_max]
        x = np.append(x, v_max)
    x_label, x_bins = pd.cut(x, bins=bins, retbins=True)
    x_label = pd.DataFrame(x_label)
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
    mid_all_sort, interval_sum_sort, left, right = cal_dist(x, bins)         # calculate distribution
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
    line = r2_line(true, predict)
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


def r2_line(y1, y2):
    y1_min, y1_max = np.min(y1), np.max(y1)
    y2_min, y2_max = np.min(y2), np.max(y2)
    y_min = np.max(np.array([y1_min, y2_min]))
    y_max = np.min(np.array([y1_max, y2_max]))
    y1_len, y2_len = y1.shape[0], y2.shape[0]
    y_len = np.max(np.array([y1_len, y2_len]))
    line = np.linspace(y_min, y_max, y_len)
    return line


def result(true, pred, loc, t, fig_si, fo_si, fo_ti_si, fo_te, c='lightcoral'):
    r2 = r2_score(true, pred)
    line = r2_line(true, pred)
    line_min, line_max = np.min(line), np.max(line)
    fig = plt.figure(figsize=fig_si)
    ax = fig.add_subplot(111)
    lim_min, lim_max = line_min - 0.2, line_max + 0.2
    ax_result(ax, line, lim_min, lim_max, r2, true, pred, loc, fo_te, c)

    plt.xlabel("True Magnitudes", fontsize=fo_si, labelpad=20)
    plt.ylabel("Estimated Magnitudes", fontsize=fo_si, labelpad=20)
    plt.title(t, fontsize=fo_si, pad=30)
    plt.xticks(fontsize=fo_ti_si)
    plt.yticks(fontsize=fo_ti_si)
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    return fig


def ax_result(ax, line, lim_min, lim_max, r2, true, pred, loc, fo_te, c):
    ax.plot(line, line, linestyle="--", c="black")
    ax.scatter(true, pred, fc='none', ec=c, s=300)
    ax.patch.set_facecolor('grey')
    ax.patch.set_alpha(0.3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(c="white", linewidth=5)
    ax.xaxis.grid(c="white", linewidth=5)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    if loc != []:
        lim_length = lim_max - lim_min
        x_loc = loc[0] * lim_length + lim_min
        y_loc = loc[1] * lim_length + lim_min
        t = ax.text(x_loc, y_loc, "$R^{2}$" + " = {:.4f}".format(r2), fontsize=fo_te)
        t.set_bbox(dict(facecolor=c, alpha=0.5, edgecolor=c))
    return ax


def dist(x, bins, jump, pos, t, fig_si, fo_si, fo_ti_si, fo_te_si, x_name, v_min=None, v_max=None,
         alpha=1, c='royalblue', y_name="Frequency", tig=False):
    fig = plt.figure(figsize=fig_si)
    ax = fig.add_subplot(111)
    mid_all_sort, interval_sum_sort, left, right = cal_dist(x, bins, v_min, v_max)
    ax_dist(ax, interval_sum_sort, x, bins, pos, t, fo_si, fo_te_si, c, alpha)

    labels = mid_all_sort[np.arange(0, bins, jump)]
    ticks = np.arange(bins)[np.arange(0, bins, jump)]
    plt.xticks(ticks=ticks, labels=labels, fontsize=fo_ti_si)
    plt.yticks(fontsize=fo_ti_si)
    if x_name is not None:
        plt.xlabel(x_name, fontsize=fo_si, labelpad=30)
    if y_name is not None:
        plt.ylabel(y_name, fontsize=fo_si, labelpad=30)
    if tig:
        plt.subplots_adjust(bottom=0.15)
    return fig


def ax_dist(ax, interval_sum_sort, x, bins, pos, t, fo_si, fo_te_si, c, alpha):
    if t is not None:
        ax.set_title(t, fontsize=fo_si, pad=30)
    ax.bar(x=np.arange(bins), height=interval_sum_sort, color=c, edgecolor="black", linewidth=2, alpha=alpha)
    ax.patch.set_facecolor('grey')
    ax.patch.set_alpha(0.3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(c="white", linewidth=5)
    ax.xaxis.grid(c="white", linewidth=5)
    if pos != []:
        mean, std = np.mean(x), np.std(x)
        axes = plt.gca()
        lim_x_min, lim_x_max = axes.get_xlim()
        lim_y_min, lim_y_max = axes.get_ylim()
        lim_x_length, lim_y_length = lim_x_max - lim_x_min, lim_y_max - lim_y_min
        x_loc = pos[0] * lim_x_length + lim_x_min
        y_loc = pos[1] * lim_y_length + lim_y_min
        t = ax.text(x_loc, y_loc, r'$\mu^{ e}$' + " = {:.4f}\n".format(mean) + r'$\sigma^{ e}$' + " = {:.4f}".format(std), fontsize=fo_te_si)
        t.set_bbox(dict(facecolor=c, alpha=0.5, edgecolor=c))
    return ax


def map_view(arr_list, t_list, pos, lat_min, lat_max, lon_min, lon_max, fig_si, fo_si, fo_ti_si):
    num = len(arr_list)
    fig = plt.figure(figsize=fig_si)
    axes = fig.subplots(1, num)
    for i in range(num):
        ax = axes[i]
        arr = arr_list[i].tolist()
        t = t_list[i]
        ax.set_title(t, fontsize=fo_si, pad=20)
        m = Basemap(ax=ax,
                    llcrnrlon=lon_min,
                    llcrnrlat=lat_min,
                    urcrnrlon=lon_max,
                    urcrnrlat=lat_max)
        # m.arcgisimage(service='World_Imagery', xpixels=800, verbose=True)
        x, y = m(pos[:, 0], pos[:, 1])
        # m.drawcoastlines()
        # m.drawmapboundary()
        parallels = np.linspace(lat_min, lat_max, 4)
        meridians = np.linspace(lon_min, lon_max, 6)[1:-1]
        m.drawparallels(np.around(parallels, 0), labels=[1, 0, 0, 0], fontsize=fo_ti_si)
        m.drawmeridians(np.around(meridians, 0), labels=[0, 0, 0, 1], fontsize=fo_ti_si)
        cs = m.scatter(x, y, marker='o', c=arr, s=50, linewidth=2, cmap=plt.cm.jet)
    fig.subplots_adjust(wspace=0.2)
    cbar = fig.colorbar(cs, ax=axes, location='right', pad=0.01, shrink=0.45)
    cbar.set_label("Magnitudes", fontsize=fo_si, labelpad=20)
    cbar.ax.tick_params(labelsize=fo_ti_si)
    return fig


def error_fea(error, fea, fig_si, fo_si, fo_ti_si, x_name=None, c='green'):
    pos = np.hstack([error.reshape(-1, 1), fea.reshape(-1, 1)])
    pos_c = gaussian_kde(pos.T)(pos.T)
    fig = plt.figure(figsize=fig_si)
    ax = fig.add_subplot(111)
    cs = ax.scatter(fea, error, c=pos_c, s=300, alpha=0.8, cmap=plt.cm.jet)
    ax.hlines(0, xmin=np.min(fea), xmax=np.max(fea), colors="black", ls='dashed', lw=6)
    ax.patch.set_facecolor('grey')
    ax.patch.set_alpha(0.3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(c="white", linewidth=5)
    ax.xaxis.grid(c="white", linewidth=5)
    cbar = fig.colorbar(cs, ax=ax, location='top', pad=0.03, shrink=0.8)
    cbar.set_label("Kernel Density", fontsize=fo_si - 5, labelpad=20)
    cbar.ax.tick_params(labelsize=fo_ti_si - 5)
    if x_name is not None:
        ax.set_xlabel(x_name, fontsize=fo_si, labelpad=30)
    ax.set_ylabel("Estimated errors", fontsize=fo_si, labelpad=20)
    # ax.xaxis.set_label_position("top")
    # ax.xaxis.tick_top()
    ax.tick_params(axis='both', labelsize=fo_ti_si)
    plt.subplots_adjust(bottom=0.15)
    return fig
