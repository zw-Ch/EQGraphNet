"""
测试（真实）
"""
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import torch
import torch.nn as nn
from obspy import read
import sys
sys.path.append("..")
import func.net as net


def load(root, dir_x, name_x, orie_x):
    x_e = read(osp.join(root, dir_x, name_x + "." + orie_x + "E")).traces[0].data[:6000]
    x_n = read(osp.join(root, dir_x, name_x + "." + orie_x + "N")).traces[0].data[:6000]
    x_z = read(osp.join(root, dir_x, name_x + "." + orie_x + "Z")).traces[0].data[:6000]
    x = np.vstack([x_e.reshape(1, -1), x_n.reshape(1, -1), x_z.reshape(1, -1)])
    return x


"""
记录，地震事件
发生时刻（event_info中唯一标识）、数据文件名、真实震级
"2016-10-30T07:01:32.630000Z"、"3A.MZ25"、4.0

"""

device = "cuda:1" if torch.cuda.is_available() else "cpu"
root = "reality_data"
dir_x = "2016_10_30"
name_x = "3A.MZ25"
orie_x = "EH"

x = load(root, dir_x, name_x, orie_x)
x = torch.from_numpy(x).float().unsqueeze(0).to(device)

"""
loading trained model
"""
re_ad = "../result/mag_predict"
sm_scale = "ml"
train_ratio = 0.75
m = 200000

m_train = int(m * train_ratio)
m_test = m - m_train
name = "chunk2"

EQG = net.EQGraphNet("gcn", "ts_un", 1, device).to(device)
EQG.load_state_dict(torch.load(osp.join(re_ad, "EQGraphNet", "model_{}_{}_{}_{}.pkl".
                                        format(sm_scale, name, m_train, m_test))))
y = EQG(x)
print("估计结果：{}".format(y.item()))


print()
plt.show()
print()
