"""
plot the estimated results of EQGraphNet
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import os.path as osp
import sys

import torch

sys.path.append("..")
import func.draw as draw
import func.net as net


def cal_class(true, pred):
    num = true.shape[0]
    cl_true = np.where(true < 0, 0, 1)
    cl_pred = np.where(pred < 0, 0, 1)

    correct = np.sum(cl_true == cl_pred)
    tp = np.sum((cl_true * cl_pred) == 1)
    fn = np.sum((cl_true * (1 - cl_pred)) == 1)
    fp = np.sum(((1 - cl_true) * cl_pred) == 1)
    tn = np.sum(((1 - cl_true) * (1 - cl_pred)) == 1)

    acc = correct / num
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * ((pre * rec) / (pre + rec))
    return cl_true, cl_pred, acc, pre, rec, f1


def mid_class(true, pred, cl_true, cl_pred):
    idx_fn = np.argwhere((cl_true == 1) & (cl_pred == 0)).reshape(-1)           # 地震，被识别为噪声
    idx_fp = np.argwhere((cl_true == 0) & (cl_pred == 1)).reshape(-1)           # 噪声，被识别为地震
    fn_mag = true[idx_fn]
    fp_mag = pred[idx_fp]
    return fn_mag, fp_mag


name = "chunk2"
m = 200000                           # number of samples
train_ratio = 0.75
m_train = int(m * train_ratio)       # number of training samples
m_test = m - m_train                 # number of testing samples
sm_scale = "ml_md"
re_ad = osp.join('../result/EQGraphNet')
loc = [0.07, 0.85]
pos = [0.07, 0.72]
fig_si_r = (26, 20)
fig_si_m = (26, 13)
fo_si = 80
fo_ti_si = 50
fo_te_si = 70
bins = 40
jump = 8
c_re = 'hotpink'
c_er = ''
den = False
save_fig = True

"""
Loading results
"""
pred = np.load(osp.join(re_ad, "test_pred_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
true = np.load(osp.join(re_ad, "test_true_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
error = pred - true

cl_true, cl_pred, acc, pre, rec, f1 = cal_class(true, pred)
print("Acc: {:.4f}  Pre: {:.4f}  Rec: {:.4f}  f1: {:.4f}".format(acc, pre, rec, f1))

fn_mag, fp_mag = mid_class(true, pred, cl_true, cl_pred)
fig_fn = draw.dist(fn_mag, 5, 1, pos, None, fig_si_m, fo_si, fo_ti_si, fo_te_si, "Errors")
fig_fp = draw.dist(fp_mag, 5, 1, pos, None, fig_si_m, fo_si, fo_ti_si, fo_te_si, "Errors")

# fig_re = draw.result(true, pred, loc, "$m_{L}$ $m_{D}$", fig_si_r, fo_si, fo_ti_si, fo_te_si, den)

print()
plt.show()
print()
