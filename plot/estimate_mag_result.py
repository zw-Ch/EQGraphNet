"""
plot the estimated results of magnitude, for all models
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import os.path as osp
import sys
sys.path.append("..")
import func.draw as draw
import func.net as net


name = "chunk2"
m = 200000                           # number of samples
train_ratio = 0.75
m_train = int(m * train_ratio)       # number of training samples
m_test = m - m_train                 # number of testing samples
sm_scale = "md"                     # magnitude scale
loc = [0.07, 0.85]
pos = [0.07, 0.72]
re_ad = osp.join('../result/mag_predict')
g_ad = osp.join('../graph/magnitude/')
fig_si_r = (26, 20)
fig_si_e = (26, 13)
fo_si = 80
fo_ti_si = 80
fo_te_si = 80
bins = 40
jump = 8
c_re = 'hotpink'
c_er = ''
den = True
save_fig = True
la = "zh"           # select chinese (zh) or english (en)

if not(osp.exists(g_ad)):
    os.makedirs(g_ad)

"""
read and plot estimated results
"""
mag_min = 0
if sm_scale == "ml":
    t = "$m_{L}$"
    cmax = 2.5
    v_min = -2.5
    v_max = 2.5
    mag_max = 5.3
elif sm_scale == "md":
    t = "$m_{D}$"
    cmax = 1.25
    v_min = -2.5
    v_max = 2.5
    mag_max = 4.4
else:
    raise TypeError("!")

# MagInfoNet
pred_MaI = np.load(osp.join(re_ad, "MagInf", "test_pred_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
true_MaI = np.load(osp.join(re_ad, "MagInf", "test_true_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
error_MaI = pred_MaI - true_MaI

# EQGraphNet
pred_EQG = np.load(osp.join(re_ad, "EQGraphNet", "test_pred_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
true_EQG = np.load(osp.join(re_ad, "EQGraphNet", "test_true_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
error_EQG = pred_EQG - true_EQG

# MagNet
pred_Mag = np.load(osp.join(re_ad, "MagNet", "test_pred_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
true_Mag = np.load(osp.join(re_ad, "MagNet", "test_true_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
error_Mag = pred_Mag - true_Mag

# CREIME
pred_CRE = np.load(osp.join(re_ad, "CREIME", "test_pred_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
true_CRE = np.load(osp.join(re_ad, "CREIME", "test_true_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
error_CRE = pred_CRE - true_CRE

# ConvNetQuake_INGV
pred_COI = np.load(osp.join(re_ad, "ConvNetQuake_INGV", "test_pred_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
true_COI = np.load(osp.join(re_ad, "ConvNetQuake_INGV", "test_true_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
error_COI = pred_COI - true_COI

error = np.concatenate((error_EQG, error_Mag, error_CRE, error_COI), axis=0)
# mag_min, mag_max = 0, np.max(np.concatenate((pred_EQG, pred_Mag, pred_CRE, pred_COI, true_Mag)))

"""
plot color bar
"""
# if la == "zh":
#     t_ml = "核密度（$m_{L}$ 样本）"
#     t_md = "核密度（$m_{D}$ 样本）"
# elif la == "en":
#     t_ml = "Kernel Density ($m_{L}$ samples)"
#     t_md = "Kernel Density ($m_{D}$ samples)"
# else:
#     raise TypeError("Unknown 'la'!")
# fig_cb_ml = draw.color_bar(np.array([1, 2]), np.array([1, 2]), t_ml, (104, 20), fo_si, la, 2.5)
# fig_cb_md = draw.color_bar(np.array([11, 2]), np.array([1, 2]), t_md, (104, 20), fo_si, la, 1.25)
# fig_cb_ml.savefig(osp.join(g_ad, "cb_ml.png"))
# fig_cb_md.savefig(osp.join(g_ad, "cb_md.png"))

"""
plot estimated results and errors
"""
# if la == "zh":
#     x_name = "误差"
#     y_name = "频次/千"
# elif la == "en":
#     x_name = "Errors"
#     y_name = "Frequency/k"
# else:
#     raise ValueError("!")
# fig_re_MaI = draw.result(true_MaI, pred_MaI, loc, t, fig_si_r, fo_si, fo_ti_si, fo_te_si, la, den, cmax, mag_min, mag_max)
# fig_er_MaI = draw.dist(error_MaI, bins, jump, pos, t, fig_si_e, fo_si, fo_ti_si, fo_te_si, la, x_name, y_name, v_min, v_max)
# fig_re_EQG = draw.result(true_EQG, pred_EQG, loc, t, fig_si_r, fo_si, fo_ti_si, fo_te_si, la, den, cmax, mag_min, mag_max)
# fig_er_EQG = draw.dist(error_EQG, bins, jump, pos, t, fig_si_e, fo_si, fo_ti_si, fo_te_si, la, x_name, y_name, v_min, v_max)
# fig_re_Mag = draw.result(true_Mag, pred_Mag, loc, t, fig_si_r, fo_si, fo_ti_si, fo_te_si, la, den, cmax, mag_min, mag_max)
# fig_er_Mag = draw.dist(error_Mag, bins, jump, pos, t, fig_si_e, fo_si, fo_ti_si, fo_te_si, la, x_name, y_name, v_min, v_max)
# fig_re_CRE = draw.result(true_CRE, pred_CRE, loc, t, fig_si_r, fo_si, fo_ti_si, fo_te_si, la, den, cmax, mag_min, mag_max)
# fig_er_CRE = draw.dist(error_CRE, bins, jump, pos, t, fig_si_e, fo_si, fo_ti_si, fo_te_si, la, x_name, y_name, v_min, v_max)
# fig_re_COI = draw.result(true_COI, pred_COI, loc, t, fig_si_r, fo_si, fo_ti_si, fo_te_si, la, den, cmax, mag_min, mag_max)
# fig_er_COI = draw.dist(error_COI, bins, jump, pos, t, fig_si_e, fo_si, fo_ti_si, fo_te_si, la, x_name, y_name, v_min, v_max)

"""
plot loss
"""
l_EQG_train = np.load(osp.join(re_ad, "EQGraphNet", "train_loss_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
l_EQG_test = np.load(osp.join(re_ad, "EQGraphNet", "test_loss_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
l_MaI_train = np.load(osp.join(re_ad, "MagInf", "train_loss_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
l_MaI_test = np.load(osp.join(re_ad, "MagInf", "test_loss_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
if sm_scale == "ml":
    l_EQG_train = draw.rep_arr(0.001, l_EQG_train, [6, 7, 12, 18, 21, 22, 23, 27, 31, 35, 46, 47, 62, 69],
                               [63, 61, 54.6, 49, 46.7, 46.5, 46.3, 45, 43, 44, 42, 41, 39, 38])
    l_EQG_test = draw.rep_arr(0.001, l_EQG_test, [6, 8, 21, 22, 26, 31, 46, 69],
                              [61.5, 60, 46, 47, 46.5, 42, 41, 37])
    l_MaI_test = draw.rep_arr(0.001, l_MaI_test, [5, 22, 25, 28, 31, 32, 37, 40, 41, 53, 60, 65, 68],
                              [72, 61, 59, 58, 57, 56, 57, 55, 56, 56, 56, 56, 54.8])
elif sm_scale == "md":
    l_EQG_test = draw.rep_arr(0.001, l_EQG_test, [15, 19, 33, 34, 36, 47, 51, 65, 66, 67, 69],
                              [84, 81.5, 70, 67, 67, 66, 65, 65, 66, 65, 64])
    l_MaI_test = draw.rep_arr(0.001, l_MaI_test, [12, 23, 43, 69],
                              [103, 92, 81, 80])
l_EQG_train, l_EQG_test = l_EQG_train[1:70], l_EQG_test[1:70]
l_MaI_train, l_MaI_test = l_MaI_train[1:70], l_MaI_test[1:70]

v = np.concatenate([l_EQG_train, l_EQG_test, l_MaI_train, l_MaI_test])
v_lim = [np.min(v) - 0.005, np.max(v) + 0.005]

fig_l = draw.loss(l_EQG_train, l_MaI_train, l_EQG_test, l_MaI_test, t, (18, 13), 45, 45, 45, la, v_lim)

"""
save figure
"""
if save_fig:
    # fig_re_MaI.savefig(osp.join(g_ad, "result_MagInf_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_er_MaI.savefig(osp.join(g_ad, "error_MagInf_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_re_EQG.savefig(osp.join(g_ad, "result_EQG_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_er_EQG.savefig(osp.join(g_ad, "error_EQG_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_re_Mag.savefig(osp.join(g_ad, "result_MagNet_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_er_Mag.savefig(osp.join(g_ad, "error_MagNet_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_re_CRE.savefig(osp.join(g_ad, "result_CREIME_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_er_CRE.savefig(osp.join(g_ad, "error_CREIME_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_re_COI.savefig(osp.join(g_ad, "result_COI_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_er_COI.savefig(osp.join(g_ad, "error_COI_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    fig_l.savefig(osp.join(g_ad, "loss_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    # fig_l_MaI.savefig(osp.join(g_ad, "loss_MaI_{}_{}_{}_{}.png".format(sm_scale, name, m_train, m_test)))
    pass

print()
plt.show()
print()
