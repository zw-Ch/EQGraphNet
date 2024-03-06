"""
plot the xai results
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
loc = [0.07, 0.85]
pos = [0.07, 0.72]
model_styles = ["EQGraphNet*", "EQLSTMNet"]
model_style = model_styles[1]
sm_scale = "md"                     # operation scale
re_ad = osp.join('../factor/xai_model_result/{}'.format(model_style))
g_ad = osp.join('../graph/xai/')
fig_si_r = (26, 20)
fig_si_e = (26, 13)
fig_si_l = (8, 4)
fo_si = 80
fo_ti_si = 50
fo_te_si = 70
bins = 40
jump = 8
c_re = 'hotpink'
c_er = ''
den = True
save_fig = True

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

# t_ml = "Kernel Density ($m_{L}$ samples)"
# fig_cb_ml = draw.color_bar(np.array([1, 2]), np.array([1, 2]), t_ml, (52, 20), fo_si, 2.5)
# t_md = "Kernel Density ($m_{D}$ samples)"
# fig_cb_md = draw.color_bar(np.array([11, 2]), np.array([1, 2]), t_md, (52, 20), fo_si, 1.25)
# fig_cb_ml.savefig(osp.join(g_ad, "cb_ml.png"))
# fig_cb_md.savefig(osp.join(g_ad, "cb_md.png"))

pred = np.load(osp.join(re_ad, "test_pred_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
true = np.load(osp.join(re_ad, "test_true_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)))
error = pred - true

fig_re = draw.result(true, pred, loc, t, fig_si_r, fo_si, fo_ti_si, fo_te_si, den, cmax, mag_min, mag_max)
fig_er = draw.dist(error, bins, jump, pos, t, fig_si_e, fo_si, fo_ti_si, fo_te_si, "Errors", v_min, v_max)

if model_style == "EQGraphNet*":
    model_style = "EQGraphNe"

if save_fig:
    fig_re.savefig(osp.join(g_ad, "result_{}_{}_{}_{}_{}.png".format(model_style, sm_scale, name, m_train, m_test)))
    fig_er.savefig(osp.join(g_ad, "error_{}_{}_{}_{}_{}.png".format(model_style, sm_scale, name, m_train, m_test)))

print()
plt.show()
print()
