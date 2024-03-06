"""
plot results of influence of adjacent nodes
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os.path as osp
import sys
sys.path.append("..")
import func.output as out


def select(df, sm_scale, gnn_style, adm_style):
    value = df[(df['gnn_style'] == gnn_style) & (df['adm_style'] == adm_style) & (df['sm_scale'] == sm_scale)]
    k = value['k'].values.astype(int)
    r2 = value['r2'].values.astype(float)
    rmse = value['rmse'].values.astype(float)
    e_mean = value['e_mean'].values.astype(float)
    e_std = value['e_std'].values.astype(float)
    return k, r2, rmse, e_mean, e_std


def ax_k_effect(ax, x_ml, x_md, y_ml, y_md, t, fo_si, fo_ti_si):
    ax.plot(x_ml, y_ml, c="blue", marker="o", ms=10, label="$m_{L}$")
    ax.plot(x_md, y_md, c="red", marker="o", ms=10, label="$m_{D}$")
    # if t == "R$^{2}$":
    #     ax.legend(fontsize=fo_si)
    ax.yaxis.grid(c="white", linewidth=2)
    # ax.xaxis.grid(c="white", linewidth=2)
    if t == "$\sigma^{\ e}$":
        ax.tick_params(axis='both', labelsize=fo_ti_si)
        ax.set_xlabel("K", fontsize=fo_si + 5, labelpad=10)
        ax.legend(fontsize=fo_si, bbox_to_anchor=(0.3, -0.1), ncol=2)
    else:
        ax.get_xaxis().set_visible(False)
        ax.tick_params(axis='y', labelsize=fo_ti_si)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7])
    ax.set_ylabel(t, fontsize=fo_si, labelpad=5)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    ax.patch.set_facecolor('grey')
    ax.patch.set_alpha(0.3)
    ax.set_axisbelow(True)
    return ax


def k_effect(k_ml, k_md, r2_ml, r2_md, rmse_ml, rmse_md, e_mean_ml, e_mean_md, e_std_ml, e_std_md, fig_si, fo_si, fo_ti_si):
    fig, axes = plt.subplots(4, 1, figsize=fig_si)
    axes[0] = ax_k_effect(axes[0], k_ml, k_md, r2_ml, r2_md, "R$^{2}$", fo_si, fo_ti_si)
    axes[1] = ax_k_effect(axes[1], k_ml, k_md, rmse_ml, rmse_md, "RMSE", fo_si, fo_ti_si)
    axes[2] = ax_k_effect(axes[2], k_ml, k_md, np.abs(e_mean_ml), np.abs(e_mean_md), "|$\mu^{\ e}$|", fo_si, fo_ti_si)
    axes[3] = ax_k_effect(axes[3], k_ml, k_md, e_std_ml, e_std_md, "$\sigma^{\ e}$", fo_si, fo_ti_si)
    fig.subplots_adjust(left=0.15)
    return fig


re_ad = "../factor/adm_result"
gnn_style = "gcn"
adm_style = "ts_un"
fig_si = (13, 20)
fo_si = 27
fo_ti_si = 27
g_ad = "../graph/operation"
save_fig = True

info = out.load_txt(osp.join(re_ad, "adm_result.txt"))
k_ml, r2_ml, rmse_ml, e_mean_ml, e_std_ml = select(info, "ml", gnn_style, adm_style)
k_md, r2_md, rmse_md, e_mean_md, e_std_md = select(info, "md", gnn_style, adm_style)

fig_ke = k_effect(k_ml, k_md, r2_ml, r2_md, rmse_ml, rmse_md, e_mean_ml, e_mean_md, e_std_ml, e_std_md, fig_si, fo_si, fo_ti_si)
if save_fig:
    fig_ke.savefig(osp.join(g_ad, "k_effect_ml_md.png"))

print()
plt.show()
print()
