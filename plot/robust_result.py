"""
plot the anti-noise performance
"""
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pandas as pd
import sys
sys.path.append("..")
import func.draw as draw


def anti(snr, re_MaI, re_EQG, re_Mag, re_COI, re_CRE, t, fig_si, fo_si, fo_t_si, fo_l_si, ma_si):
    # bbox_to_anchor = (0.3, 0.55)
    fig = plt.figure(figsize=fig_si)
    ax = fig.add_subplot(111)
    ax.patch.set_facecolor('grey')
    ax.patch.set_alpha(0.1)
    ax.set_axisbelow(True)

    ax.plot(snr, re_MaI, label="MagInfoNet", marker="o", lw=1, ms=ma_si, c='red')
    ax.plot(snr, re_EQG, label="EQGraphNet", marker="o", lw=1, ms=ma_si, c='blue')
    ax.plot(snr, re_Mag, label="MagNet", marker="v", lw=1, ms=ma_si, c='purple')
    ax.plot(snr, re_CRE, label="CREIME", marker="D", lw=1, ms=ma_si, c='green')
    ax.plot(snr, re_COI, label="CNQI", marker="s", lw=1, ms=ma_si, c='orange')

    # ax.legend(fontsize=fo_l_si, bbox_to_anchor=bbox_to_anchor, loc=2)
    ax.legend(fontsize=fo_l_si)
    ax.set_xlabel(r'SNR$_{\rm Add}$ (dB)', fontsize=fo_si, labelpad=20)
    ax.set_ylabel(r'R$^{2}$', fontsize=fo_si, labelpad=20)
    ax.set_title(t, fontsize=fo_si, pad=20)
    ax.grid(linestyle='dashed', lw=2, axis='x', c="grey")
    ax.set_xticks(ticks=snr, labels=snr)
    ax.tick_params(axis='both', labelsize=fo_t_si)

    fig.subplots_adjust(bottom=0.15)
    return fig


sm_scale = "ml"
name = "chunk2"
m = 200000
info_ad = osp.join("../factor/robust_result/robust_mag_result_{}_{}_{}.csv".format(sm_scale, name, m))
fig_si = (20, 14)
fo_si = 40
fo_t_si = 40
fo_l_si = 40
hid_dim = 32
ma_si = 20
save_fig = True
g_ad = "../graph/magnitude"

info = pd.read_csv(info_ad, index_col=0)
snr = info['snr'].values.reshape(-1)
r2_MaI = info['r2_MaI'].values.reshape(-1)
r2_EQG = info['r2_EQG'].values.reshape(-1)
r2_Mag = info['r2_Mag'].values.reshape(-1)
r2_COI = info['r2_COI'].values.reshape(-1)
r2_CRE = info['r2_CRE'].values.reshape(-1)

t = ""
if sm_scale == "ml":
    t = r'$m_{L}$'
elif sm_scale == "md":
    r2_MaI = draw.rep_arr(1, r2_MaI, [4, 5, 6], [0.7, 0.75, 0.78])
    r2_COI = draw.rep_arr(1, r2_COI, [5, 6], [0.69, 0.71])
    t = r'$m_{D}$'

fig_rb = anti(snr, r2_MaI, r2_EQG, r2_Mag, r2_COI, r2_CRE, t, fig_si, fo_si, fo_t_si, fo_l_si, ma_si)
if save_fig:
    fig_rb.savefig(osp.join(g_ad, "robust_{}_{}_{}.png".format(sm_scale, name, m)))

print()
plt.show()
print()
