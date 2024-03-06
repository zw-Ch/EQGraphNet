"""
Explainable Artificial Intelligence
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
from scipy.stats import gaussian_kde
from matplotlib.ticker import FuncFormatter
import sys
sys.path.append("..")
import func.process as pro
import func.net as net
import func.draw as draw


def plot_dc(arr_list, t_list, c_list, num_row, num_col, la, fig_si, fo_si, fo_ti_si):
    def fmt(x, pos):
        return '%1.1f' % (x * 1e5)

    formatter = FuncFormatter(fmt)
    num = len(arr_list)
    if num != num_row * num_col:
        raise ValueError("!")
    fig, axes = plt.subplots(num_row, num_col, figsize=fig_si)
    if la == "zh":
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        x_name, y_name = "第一主成分", "第二主成分"
        t_cb = r'核密度/10$^{6}$'
    elif la == "en":
        x_name, y_name = "First Component", "Second Component"
        t_cb = r'Kernel Density/10$^{6}$'
    else:
        raise TypeError("Unknown type of 'la'!")
    for i in range(num):
        cs = axes[i].scatter(arr_list[i][:, 0], arr_list[i][:, 1], c=c_list[i], s=3, cmap=plt.cm.rainbow)
        axes[i].tick_params(axis='both', labelsize=fo_ti_si)
        axes[i].set_xlabel(x_name, fontsize=fo_si, labelpad=10)
        axes[i].set_ylabel(y_name, fontsize=fo_si, labelpad=10)
        axes[i].set_title(t_list[i], fontsize=fo_si, pad=25)
    cbar = fig.colorbar(cs, ax=axes.ravel().tolist(), location='bottom', pad=0.1, shrink=0.9, aspect=80)
    cbar.set_label(t_cb, fontsize=fo_si, labelpad=10)
    cbar.ax.tick_params(labelsize=fo_si)
    cbar.ax.xaxis.set_major_formatter(formatter)
    fig.subplots_adjust(wspace=0.4, bottom=0.4)
    return fig


def get_gau_kde(x, re_ad, name):
    x_gau_kde_ad = osp.join(re_ad, "gau_kde_{}.npy".format(name))
    if osp.exists(x_gau_kde_ad):
        x_gau_kde = np.load(x_gau_kde_ad)
        return x_gau_kde
    x_gau_kde = gaussian_kde(x.T)(x.T)
    np.save(x_gau_kde_ad, x_gau_kde)
    return x_gau_kde


name = "chunk2"
sm_scale = "ml"
loc = [0.07, 0.85]
pos = [0.07, 0.72]
dc_style = "tsne"
ad_MaI = "../result/mag_predict/MagInf"
ad_EQG = "../result/mag_predict/EQGraphNet"
g_ad = "../graph/xai"
m = 200000
train_ratio = 0.75
m_train = int(m * train_ratio)
m_test = m - m_train
fig_si = (26, 20)
fo_si = 80
fo_ti_si = 80
fo_te_si = 80
bins = 40
save_fig = True
la = "zh"
t_x, t_hx = "", ""

mag_min, mag_max, v_min, v_max, den = 0, None, None, None, True
t_h, t_hg, t_rc, t_gl, sm_h, sm_hg, sm_rc, sm_gl = "", "", "", "", "", "", "", ""
if sm_scale == "ml":
    sm_h, sm_hg, sm_rc, sm_gl = "$m_{L}$", "$m_{L}$", "$m_{L}$", "$m_{L}$"
    v_min, v_max, c_max, mag_max = -2.5, 2.5, 2.5, 5.3
elif sm_scale == "md":
    sm_h, sm_hg, sm_rc, sm_gl = "$m_{D}$", "$m_{D}$", "$m_{D}$", "$m_{D}$"
    v_min, v_max, c_max, mag_max = -2.5, 2.5, 1.25, 4.4

"""
The role of RM1
"""
x = np.load(osp.join(ad_MaI, "dc_x_{}.npy".format(dc_style)))
hx = np.load(osp.join(ad_MaI, "dc_hx_{}.npy".format(dc_style)))

c_x = get_gau_kde(x, ad_MaI, "x")
c_hx = get_gau_kde(hx, ad_MaI, "hx")

if la == "zh":
    t_x, t_hx = "$X$ 降维结果", "$h_{x}$ 降维结果"
elif la == "en":
    t_x, t_hx = "The dist of $X$", "The dist of $h_{x}$"

arr_list, c_list, t_list = [x, hx], [c_x, c_hx], [t_x, t_hx]
num_row, num_col = 1, 2
fig_ts = plot_dc(arr_list, t_list, c_list, num_row, num_col, la, (19, 13), 25, 25)

"""
The role of UniMP and RM2
"""
# h_true = np.load(osp.join(ad_MaI, "PreInform/test_true_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
# h_pred = np.load(osp.join(ad_MaI, "PreInform/test_pred_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
# h_e = h_true - h_pred
# hg_true = np.load(osp.join(ad_MaI, "UniMP/test_true_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
# hg_pred = np.load(osp.join(ad_MaI, "UniMP/test_pred_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
# hg_e = hg_true - hg_pred
#
# if la == "zh":
#     t_h = r'无UniMP与RM$_{2}$'
#     t_hg = '无UniMP'
# elif la == "en":
#     t_h = r'Without UniMP and RM$_{2}$'
#     t_hg = 'Without UniMP'
#
# arr_list, t_list = [[h_true, h_pred], [hg_true, hg_pred]], [t_h, t_hg]
# fig_re_h = draw.result(h_true, h_pred, loc, sm_h, fig_si, fo_si, fo_ti_si, fo_te_si, la, t_h, den, None, mag_min, mag_max)
# fig_di_h = draw.dist(h_e, 40, 8, pos, sm_h, (26, 13), fo_si, fo_ti_si, fo_te_si, la, v_min, v_max)
# fig_re_hg = draw.result(hg_true, hg_pred, loc, sm_hg, fig_si, fo_si, fo_ti_si, fo_te_si, la, t_hg, den, None, mag_min, mag_max)
# fig_di_hg = draw.dist(hg_e, 40, 8, pos, sm_hg, (26, 13), fo_si, fo_ti_si, fo_te_si, la, v_min, v_max)

"""
The role of Residual Connection and GCN Layer
"""
# h_rc_true = np.load(osp.join(ad_EQG, "EQGraphNe/test_true_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
# h_rc_pred = np.load(osp.join(ad_EQG, "EQGraphNe/test_pred_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
# h_rc_e = h_rc_true - h_rc_pred
# h_gl_true = np.load(osp.join(ad_EQG, "EQLSTMNet/test_true_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
# h_gl_pred = np.load(osp.join(ad_EQG, "EQLSTMNet/test_pred_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
# h_gl_e = h_gl_true - h_gl_pred
#
# if la == "zh":
#     t_rc = '无残差连接'
#     t_gl = '使用Bi-LSTM代替GCN'
# elif la == "en":
#     t_rc = 'Without Residual Connection'
#     t_gl = 'Using Bi-LSTM replace GCN'
#
# fig_re_rc = draw.result(h_rc_true, h_rc_pred, loc, sm_rc, fig_si, fo_si, fo_ti_si, fo_te_si, la, t_rc, den, None, mag_min, mag_max)
# fig_di_rc = draw.dist(h_rc_e, 40, 8, pos, sm_rc, (26, 13), fo_si, fo_ti_si, fo_te_si, la, v_min, v_max)
# fig_re_gl = draw.result(h_gl_true, h_gl_pred, loc, sm_gl, fig_si, fo_si, fo_ti_si, fo_te_si, la, t_gl, den, None, mag_min, mag_max)
# fig_di_gl = draw.dist(h_gl_e, 40, 8, pos, sm_gl, (26, 13), fo_si, fo_ti_si, fo_te_si, la, v_min, v_max)

if save_fig:
    fig_ts.savefig(osp.join(g_ad, "template_select.png"))
    # fig_re_h.savefig(osp.join(g_ad, "result_PreInform_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_di_h.savefig(osp.join(g_ad, "error_PreInform_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_re_hg.savefig(osp.join(g_ad, "result_UniMP_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_di_hg.savefig(osp.join(g_ad, "error_UniMP_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_re_rc.savefig(osp.join(g_ad, "result_EQGraphNe_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_di_rc.savefig(osp.join(g_ad, "error_EQGraphNe_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_re_gl.savefig(osp.join(g_ad, "result_EQLSTMNet_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_di_gl.savefig(osp.join(g_ad, "error_EQLSTMNet_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    pass

print()
plt.show()
print()
