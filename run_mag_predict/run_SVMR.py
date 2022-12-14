"""
Magnitude Prediction
[Fast magnitude determination using a single seismological station record implementing machine learning techniques]
"""
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
from torch.utils.data import DataLoader
import sys
sys.path.append('..')
import func.process as pro
import func.net as net
import func.draw as draw


train_ratio = 0.75                  # ratio of training samples
m = 10000                           # number of samples
sm_scale = "ml"                     # magnitude scale
save_txt = True
save_model = True
save_np = True
save_fig = True
random = False
fig_si = (12, 12)          # The size of figures
fo_si = 40
fo_ti_si = 30
bins = 40
jump = 8

re_ad = osp.join("../result/mag_predict", "SVMR")
if not(osp.exists(re_ad)):
    os.makedirs(re_ad)
