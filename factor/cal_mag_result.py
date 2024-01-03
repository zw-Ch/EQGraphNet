"""
calculate the estimated results of magnitude, for all models
"""
import numpy as np
import os.path as osp
import sys
sys.path.append("..")
import func.net as net


def mag_ran(true, pred, v_min, v_max, style):
    idx = np.argwhere((true >= v_min) & (true <= v_max)).reshape(-1)
    if idx.shape[0] == 0:
        raise ValueError("The range of magnitude is too small!")
    true_remain, pred_remain = true[idx], pred[idx]
    r2 = net.cal_r2_one_arr(true_remain, pred_remain)
    rmse = net.cal_rmse_one_arr(true_remain, pred_remain)
    if style == "rmse":
        return rmse
    elif style == "r2":
        return r2
    else:
        raise TypeError("Unknown Type of 'style'!")


name = "chunk2"
m = 200000                           # number of samples
train_ratio = 0.75
m_train = int(m * train_ratio)       # number of training samples
m_test = m - m_train                 # number of testing samples
sm_scale = "ml"                     # magnitude scale
style = "rmse"
re_ad = osp.join('../result/mag_predict')
save_fig = True

"""
read and calculate estimated results
"""
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

# print the metrics results
r2_EQG, rmse_EQG = net.cal_r2_one_arr(true_EQG, pred_EQG), net.cal_rmse_one_arr(true_EQG, pred_EQG)
print("EQG:  r2 = {:.4f}  rmse = {:.4f}".format(r2_EQG, rmse_EQG))
r2_Mag, rmse_Mag = net.cal_r2_one_arr(true_Mag, pred_Mag), net.cal_rmse_one_arr(true_Mag, pred_Mag)
print("Mag:  r2 = {:.4f}  rmse = {:.4f}".format(r2_Mag, rmse_Mag))
r2_CRE, rmse_CRE = net.cal_r2_one_arr(true_CRE, pred_CRE), net.cal_rmse_one_arr(true_CRE, pred_CRE)
print("CRE:  r2 = {:.4f}  rmse = {:.4f}".format(r2_CRE, rmse_CRE))
r2_COI, rmse_COI = net.cal_r2_one_arr(true_COI, pred_COI), net.cal_rmse_one_arr(true_COI, pred_COI)
print("COI:  r2 = {:.4f}  rmse = {:.4f}".format(r2_COI, rmse_COI))

"""
results in different range
"""
print("\n{}, results: ".format(style))
print("0 ~ 1:  EQG = {:.4f}  Mag = {:.4f}  CRE = {:.4f}  COI = {:.4f}".format(
    mag_ran(true_EQG, pred_EQG, 0, 1, style), mag_ran(true_Mag, pred_Mag, 0, 1, style),
    mag_ran(true_CRE, pred_CRE, 0, 1, style), mag_ran(true_COI, pred_COI, 0, 1, style)))
print("1 ~ 2:  EQG = {:.4f}  Mag = {:.4f}  CRE = {:.4f}  COI = {:.4f}".format(
    mag_ran(true_EQG, pred_EQG, 1, 2, style), mag_ran(true_Mag, pred_Mag, 1, 2, style),
    mag_ran(true_CRE, pred_CRE, 1, 2, style), mag_ran(true_COI, pred_COI, 1, 2, style)))
print("2 ~ 3:  EQG = {:.4f}  Mag = {:.4f}  CRE = {:.4f}  COI = {:.4f}".format(
    mag_ran(true_EQG, pred_EQG, 2, 3, style), mag_ran(true_Mag, pred_Mag, 2, 3, style),
    mag_ran(true_CRE, pred_CRE, 2, 3, style), mag_ran(true_COI, pred_COI, 2, 3, style)))
if sm_scale == "ml":
    print("3 ~ 4:  EQG = {:.4f}  Mag = {:.4f}  CRE = {:.4f}  COI = {:.4f}".format(
        mag_ran(true_EQG, pred_EQG, 3, 4, style), mag_ran(true_Mag, pred_Mag, 3, 4, style),
        mag_ran(true_CRE, pred_CRE, 3, 4, style), mag_ran(true_COI, pred_COI, 3, 4, style)))
    print(" >= 4:  EQG = {:.4f}  Mag = {:.4f}  CRE = {:.4f}  COI = {:.4f}".format(
        mag_ran(true_EQG, pred_EQG, 4, 6, style), mag_ran(true_Mag, pred_Mag, 4, 6, style),
        mag_ran(true_CRE, pred_CRE, 4, 6, style), mag_ran(true_COI, pred_COI, 4, 6, style)))
elif sm_scale == "md":
    print(" >= 3:  EQG = {:.4f}  Mag = {:.4f}  CRE = {:.4f}  COI = {:.4f}".format(
        mag_ran(true_EQG, pred_EQG, 3, 6, style), mag_ran(true_Mag, pred_Mag, 3, 6, style),
        mag_ran(true_CRE, pred_CRE, 3, 6, style), mag_ran(true_COI, pred_COI, 3, 6, style)))

"""
parameter number
"""
n_MaI = 2 * (2 * 1000 + 1000 + 1000 * 6000 + 6000)

print()
