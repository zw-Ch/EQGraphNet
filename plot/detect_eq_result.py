"""
plot the detected results of earthquake
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import os.path as osp
import sys
sys.path.append("..")
import func.draw as draw


m_no, m_eq = 200000, 200000
train_ratio = 0.75
m_no_train, m_eq_train = int(m_no * train_ratio), int(m_eq * train_ratio)
m_no_test, m_eq_test = m_no - m_no_train, m_eq - m_eq_train
re_ad = osp.join('../result/eq_detect/EqDetect')
g_ad = osp.join('../graph/detect_earthquake')
fig_si = (5, 4.8)
fo_si = 17
fo_ti_si = 15
save_fig = True

"""
read results
"""
acc = np.load(osp.join(re_ad, "acc_test_{}_{}_{}_{}.npy".format(m_no_train, m_no_test, m_eq_train, m_eq_test)))
pre = np.load(osp.join(re_ad, "pre_test_{}_{}_{}_{}.npy".format(m_no_train, m_no_test, m_eq_train, m_eq_test)))
rec = np.load(osp.join(re_ad, "rec_test_{}_{}_{}_{}.npy".format(m_no_train, m_no_test, m_eq_train, m_eq_test)))
f1 = np.load(osp.join(re_ad, "f1_test_{}_{}_{}_{}.npy".format(m_no_train, m_no_test, m_eq_train, m_eq_test)))

fig_metric = plt.figure(figsize=fig_si)
plt.plot(pre, lw=2, label="Precision", c="dodgerblue")
plt.plot(rec, lw=2, label="Recall", c="limegreen")
plt.plot(f1, lw=2, label="F1-score", c="r")
plt.legend(fontsize=fo_si)
plt.xlabel("Epoch", fontsize=fo_si)
plt.ylabel("Value", fontsize=fo_si)
plt.xticks(fontsize=fo_ti_si)
plt.yticks(fontsize=fo_ti_si)
fig_metric.subplots_adjust(left=0.2, bottom=0.15)

if save_fig:
    fig_metric.savefig(osp.join(g_ad, "metric.png"))

print()
plt.show()
print()
