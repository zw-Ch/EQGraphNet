"""
From the view of map
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
import sys
sys.path.append('..')
import func.draw as draw
import func.output as out


sm_list = ["ml", "md"]
name = "chunk2"
root = "/home/chenziwei2021/standford_dataset/{}".format(name)
m = 200000                      # number of data
re_ad = "../result/mag_predict"
train_ratio = 0.75
with_metric = False             # whether plot regressive metric in title

# [17, 20, -69, -64], [57, 60, -156, -151], [46, 49, -126, -121], [18.8, 20, -156.5, -154.5]
lat_min, lat_max, lon_min, lon_max = 18.8, 20, -156.5, -154.5

vmax = 5.3                      # display highest operation
m_train = int(m * train_ratio)
m_test = m - m_train
fig_si = (25, 8)
fo_si = 18
fo_ti_si = 18
save_fig = True
g_ad = "../graph/map"

pos, true, trace, pred_MaI, pred_EQG, pred_Mag, pred_CRE, pred_COI = out.read_sm(sm_list, re_ad, name, m_train, m_test)
pos, true, trace, pred_MaI, pred_EQG, pred_Mag, pred_CRE, pred_COI = out.select_range(
    pos, true, trace, lat_min, lat_max, lon_min, lon_max, pred_MaI, pred_EQG, pred_Mag, pred_CRE, pred_COI)

print("MaI: min = {:.4f}  max = {:.4f}".format(np.min(pred_MaI), np.max(pred_MaI)))
print("EQG: min = {:.4f}  max = {:.4f}".format(np.min(pred_EQG), np.max(pred_EQG)))
print("Mag: min = {:.4f}  max = {:.4f}".format(np.min(pred_Mag), np.max(pred_Mag)))
print("CRE: min = {:.4f}  max = {:.4f}".format(np.min(pred_CRE), np.max(pred_CRE)))
print("COI: min = {:.4f}  max = {:.4f}".format(np.min(pred_COI), np.max(pred_COI)))
print()

r2_MaI = out.cal_r2_one_arr(true, pred_MaI)
r2_EQG = out.cal_r2_one_arr(true, pred_EQG)
r2_Mag = out.cal_r2_one_arr(true, pred_Mag)
r2_CRE = out.cal_r2_one_arr(true, pred_CRE)
r2_COI = out.cal_r2_one_arr(true, pred_COI)
rmse_MaI = out.cal_rmse_one_arr(true, pred_MaI)
rmse_EQG = out.cal_rmse_one_arr(true, pred_EQG)
rmse_Mag = out.cal_rmse_one_arr(true, pred_Mag)
rmse_CRE = out.cal_rmse_one_arr(true, pred_CRE)
rmse_COI = out.cal_rmse_one_arr(true, pred_COI)
print("MaI:  r2 = {:.4f}  rmse = {:.4f}".format(r2_MaI, rmse_MaI))
print("EQG:  r2 = {:.4f}  rmse = {:.4f}".format(r2_EQG, rmse_EQG))
print("Mag:  r2 = {:.4f}  rmse = {:.4f}".format(r2_Mag, rmse_Mag))
print("CRE:  r2 = {:.4f}  rmse = {:.4f}".format(r2_CRE, rmse_CRE))
print("COI:  r2 = {:.4f}  rmse = {:.4f}".format(r2_COI, rmse_COI))

# """
# plot figure simply
# """
# fig_ms, axes = plt.subplots(1, 5, figsize=fig_si)
# axes[0].scatter(pos[:, 0], pos[:, 1], c=true)
# axes[0].set_title("true", fontsize=fo_si)
# axes[1].scatter(pos[:, 0], pos[:, 1], c=pred_EQG)
# axes[1].set_title("pred EQG", fontsize=fo_si)
# axes[2].scatter(pos[:, 0], pos[:, 1], c=pred_Mag)
# axes[2].set_title("pred Mag", fontsize=fo_si)
# axes[3].scatter(pos[:, 0], pos[:, 1], c=pred_COI)
# axes[3].set_title("pred COI", fontsize=fo_si)
# axes[4].scatter(pos[:, 0], pos[:, 1], c=pred_CRE)
# axes[4].set_title("pred CRE", fontsize=fo_si)

"""
Earthquake characterization
"""
# df = pd.read_csv(osp.join(root, name + ".csv"))
# df = out.select_trace_small(trace, df)

# output the earthquake information by given trace
df = out.get_eq_info(root, name, trace.reshape(-1))
df.to_csv(osp.join(g_ad, "{}_lat_{}_{}_lon_{}_{}.csv".format(name, lat_min, lat_max, lon_min, lon_max)))

"""
plot results on map
"""
arr_list = [true, pred_MaI, pred_EQG, pred_Mag, pred_CRE, pred_COI]
if with_metric:
    t_MaI = "r2 = {:.4f}  rmse = {:.4f}\nMagInfoNet".format(r2_MaI, rmse_MaI)
    t_EQG = "r2 = {:.4f}  rmse = {:.4f}\nEQGraphNet".format(r2_EQG, rmse_EQG)
    t_Mag = "r2 = {:.4f}  rmse = {:.4f}\nMagNet".format(r2_Mag, rmse_Mag)
    t_CRE = "r2 = {:.4f}  rmse = {:.4f}\nCREIME".format(r2_CRE, rmse_CRE)
    t_COI = "r2 = {:.4f}  rmse = {:.4f}\nCNQI".format(r2_COI, rmse_COI)
else:
    t_MaI = "MagInfoNet"
    t_EQG = "EQGraphNet"
    t_Mag = "MagNet"
    t_CRE = "CREIME"
    t_COI = "CNQI"
t_list = ["True", t_MaI, t_EQG, t_Mag, t_CRE, t_COI]
fig_mv = draw.map_view(arr_list, t_list, pos, lat_min, lat_max, lon_min, lon_max, fig_si, fo_si, fo_ti_si, None, vmax)

# """
# plot errors on map
# """
# error_EQG = pred_EQG - true
# error_Mag = pred_Mag - true
# error_COI = pred_COI - true
# error_CRE = pred_CRE - true
# arr_list = [error_EQG, error_Mag, error_COI, error_CRE]
# t_EQG = "r2 = {:.4f}  rmse = {:.4f}\nEQGraphNet".format(r2_EQG, rmse_EQG)
# t_Mag = "r2 = {:.4f}  rmse = {:.4f}\nMagNet".format(r2_Mag, rmse_Mag)
# t_COI = "r2 = {:.4f}  rmse = {:.4f}\nConvNetQuake_INGV".format(r2_COI, rmse_COI)
# t_CRE = "r2 = {:.4f}  rmse = {:.4f}\nCREIME".format(r2_CRE, rmse_CRE)
# t_list = [t_EQG, t_Mag, t_COI, t_CRE]
# fig_mve = draw.map_view(arr_list, t_list, pos, lat_min, lat_max, lon_min, lon_max, fig_si, fo_si, fo_ti_si)

if save_fig:
    # fig_ms.savefig(osp.join(g_ad, "map_view_{}_{}_lat_{}_{}_lon_{}_{}_simple.png".
    #                         format(name, m, lat_min, lat_max, lon_min, lon_max)))
    fig_mv.savefig(osp.join(g_ad, "mv_{}_{}_lat_{}_{}_lon_{}_{}.png".
                            format(name, m, lat_min, lat_max, lon_min, lon_max)))
    # fig_mve.savefig(osp.join(g_ad, "mve_{}_{}_lat_{}_{}_lon_{}_{}.png".
    #                          format(name, m, lat_min, lat_max, lon_min, lon_max)))

print()
plt.show()
print()
