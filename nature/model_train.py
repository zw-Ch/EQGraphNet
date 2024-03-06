"""
training models on natural earthquake signals
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
from torch.utils.data import DataLoader
import sys
sys.path.append('..')
import func.nature as nat
import func.net as net


device = "cuda:1" if torch.cuda.is_available() else "cpu"
lr = 0.0005
weight_decay = 0.0005
batch_size = 64
epochs = 100
adm_style = "ts_un"
gnn_style = "gcn"
k = 1
train_ratio = 0.75
wav_style = "arr"

re_ad_EQG = osp.join("../result/mag_predict", "EQGraphNet")
EQG = net.EQGraphNet(gnn_style, adm_style, k, device).to(device)
EQG.load_state_dict(torch.load(osp.join(re_ad_EQG, "model_ml_chunk2_150000_50000.pkl")))

true, pred = nat.evaluate(EQG, device, "/home/chenziwei2021/pyn/other_codes/ASSCUDGNN/data", wav_style)

idx = np.argwhere((pred >= 3) & (pred <= 5)).reshape(-1)
true, pred = true[idx], pred[idx]

plt.figure()
plt.scatter(true, pred)

print()
plt.show()
print()
