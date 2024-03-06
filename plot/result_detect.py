"""
plot results of earthquake detection
"""
import seaborn as sn
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path as osp
from torch.utils.data import DataLoader
from sklearn import metrics
import sys
sys.path.append('..')
import func.process as pro
import func.net as net
from func.net import EqDetect


device = "cuda:1" if torch.cuda.is_available() else "cpu"
batch_size = 256
adm_style = "ts_un"
gnn_style = "gcn"
k = 1
train_ratio = 0.75
re_ad = "../result/eq_detect/EqDetect"
fig_si = (8, 8)
fo_si = 17
fo_ti_si = 15
save_fig = True
g_ad = "../graph/detect_earthquake"

m_no, m_eq = 200000, 200000                                                       # number of samples
m_no_train, m_eq_train = int(m_no * train_ratio), int(m_eq * train_ratio)       # number of training samples
m_no_test, m_eq_test = m_no - m_no_train, m_eq - m_eq_train                     # number of testing samples
name_no, name_eq = "chunk1", "chunk2"
root_no = "/home/chenziwei2021/standford_dataset/{}".format(name_no)
root_eq = "/home/chenziwei2021/standford_dataset/{}".format(name_eq)

idx_train_no, idx_test_no = pro.get_train_or_test_idx(m_no, m_no_train)
no_test = pro.Chunk(m_no, False, m_no_train, idx_test_no, root_no, name_no)
idx_train_eq, idx_test_eq = pro.get_train_or_test_idx(m_eq, m_eq_train)
eq_test = pro.Chunk(m_eq, False, m_eq_train, idx_test_eq, root_eq, name_eq)
no_test_data, eq_test_data = no_test.data, eq_test.data

test_data = torch.cat((no_test_data, eq_test_data), dim=0)
test_label = torch.cat((torch.ones(m_no_test), torch.zeros(m_eq_test)), dim=0).float()
test_dataset = pro.SelfData(test_data, test_label)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = net.EqDetect(gnn_style, adm_style, k, device).to(device)
model.load_state_dict(torch.load(osp.join(re_ad, "model_{}_{}_{}_{}.pkl".format(
    m_no_train, m_no_test, m_eq_train, m_eq_test))))
criterion = torch.nn.BCELoss().to(device)

size_test = len(test_dataset)
test_tp, test_tn, test_fp, test_fn = 0, 0, 0, 0
correct_test = 0
test_pred, test_true = [], []

for idx, (x_test, y_test, _) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.view(-1).to(device)
    output_test = model(x_test)
    loss_test = criterion(output_test, y_test)
    pred_test = torch.round(output_test)
    correct_test += (pred_test == y_test).type(torch.float).sum().item()

    test_tp += (pred_test * y_test == 1).sum().item()
    test_fn += ((1 - pred_test) * y_test == 1).sum().item()
    test_fp += (pred_test * (1 - y_test) == 1).sum().item()
    test_tn += ((1 - pred_test) * (1 - y_test) == 1).sum().item()

    test_pred_one = pred_test.detach().cpu().numpy().reshape(-1)
    test_true_one = y_test.detach().cpu().numpy().reshape(-1)
    if idx == 0:
        test_pred = test_pred_one
        test_true = test_true_one
    else:
        test_pred = np.concatenate((test_pred, test_pred_one), axis=0)
        test_true = np.concatenate((test_true, test_true_one), axis=0)

test_acc = correct_test / size_test
test_pre = test_tp / (test_tp + test_fp)
test_rec = test_tp / (test_tp + test_fn)
test_f1 = 2 * ((test_pre * test_rec) / (test_pre + test_rec))
print("Acc: {:.4f}  Pre: {:.4f}  Rec: {:.4f}  f1:  {:.4f}".format(test_acc, test_pre, test_rec, test_f1))
print("TP: {}  FN: {}  FP: {}  TN: {}".format(test_tp, test_fn, test_fp, test_tn))

labels = ["Earthquake", "Noise"]
fig, ax = plt.subplots()
cm = metrics.confusion_matrix(test_true, test_pred)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
sn.set(font_scale=1.4)
s = sn.heatmap(df_cm, annot=True, fmt='.5g', ax=ax)
ax.tick_params(labelsize=fo_ti_si)

s.set_xlabel('Predicted Labels', labelpad=5, fontsize=fo_si)
s.set_ylabel('True Labels', labelpad=5, fontsize=fo_si)
fig.subplots_adjust(bottom=0.2)
plt.savefig(osp.join(g_ad, "result_cm.png"))

print()
plt.show()
print()
