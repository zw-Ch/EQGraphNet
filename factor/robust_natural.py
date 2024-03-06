"""
Anti-Noise performance, the input noise is nature signal
"""
import torch
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import os
import os.path as osp
from torch.utils.data import DataLoader
import sys
sys.path.append('..')
import func.process as pro
import func.net as net


def sort_values(values):
    idx = values[:, 0].astype(int)
    idx_sort = np.argsort(idx)
    values_sort = values[idx_sort, :]
    return values_sort


def get_noise_one_natural(x, n, snr):
    e_x = np.mean(np.square(x))
    e_n = np.mean(np.square(n))
    if e_n < 0.1:                               # 有些噪声全为零值，用高斯白噪声代替
        n = np.random.normal(size=x.shape)
        e_n = np.mean(np.square(n))
    ratio = np.sqrt(e_x / e_n / snr)
    n_ = n * ratio
    n_ = np.expand_dims(n_, axis=0)
    return n_


def get_noise_natural(x, n, snr, path, sm_scale, train, save=False):
    if train:
        save_path = osp.join(path, "x_n_{}_{}_train.npy".format(sm_scale, snr))
    else:
        save_path = osp.join(path, "x_n_{}_{}_test.npy".format(sm_scale, snr))
    if osp.exists(save_path):
        x_n = np.load(save_path)
        return torch.from_numpy(x_n).float()

    if torch.is_tensor(x):
        x = x.numpy()
    if torch.is_tensor(n):
        n = n.numpy()
    x_n = copy.deepcopy(x)
    num_sample = x_n.shape[0]
    for i in range(num_sample):
        x_one, n_one = x[i, :, :], n[i, :, :]
        x_one_1, x_one_2, x_one_3 = x_one[0, :], x_one[1, :], x_one[2, :]
        n_one_1, n_one_2, n_one_3 = n_one[0, :], n_one[1, :], n_one[2, :]
        n_one_1 = get_noise_one_natural(x_one_1, n_one_1, snr)
        n_one_2 = get_noise_one_natural(x_one_2, n_one_2, snr)
        n_one_3 = get_noise_one_natural(x_one_3, n_one_3, snr)
        n_one = np.concatenate((n_one_1, n_one_2, n_one_3), axis=0)
        n_one = np.expand_dims(n_one, axis=0)
        x_n[i, :, :] = x_n[i, :, :] + n_one
    if save:
        np.save(save_path, x_n)
    return torch.from_numpy(x_n).float()


snr = 10
device = "cuda:1" if torch.cuda.is_available() else "cpu"
batch_size = 64
train_ratio = 0.75
m = 200000
sm_scale = "ml"
random = False
save_txt = True
re_ad = "../result/mag_predict"
save_ad = "../factor/robust_natural_result"
if not(osp.exists(save_ad)):
    os.makedirs(save_ad)

"""
Selection of noise and earthquake signals
"""
m_train = int(m * train_ratio)       # number of training samples
m_test = m - m_train                 # number of testing samples
name_no = "chunk1"
root_no = "/home/chenziwei2021/standford_dataset/{}".format(name_no)
name_eq = "chunk2"
root_eq = "/home/chenziwei2021/standford_dataset/{}".format(name_eq)

if not random:
    np.random.seed(100)
idx_train, idx_test = pro.get_train_or_test_idx(m, m_train)

Eq_train = pro.Chunk(m, True, m_train, idx_train, root_eq, name_eq)
Eq_test = pro.Chunk(m, False, m_train, idx_test, root_eq, name_eq)
df_train, df_test = Eq_train.df, Eq_test.df
x_train, x_test = Eq_train.data.float(), Eq_test.data.float()
sm_train = torch.from_numpy(df_train["source_magnitude"].values.reshape(-1)).float()
sm_test = torch.from_numpy(df_test["source_magnitude"].values.reshape(-1)).float()

# Select samples according to Magnitude Type
data_train, sm_train, df_train, _ = pro.remain_sm_scale(x_train, df_train, sm_train, sm_scale)
data_test, sm_test, df_test, _ = pro.remain_sm_scale(x_test, df_test, sm_test, sm_scale)

# get natural noise
No_train = pro.Chunk(m, True, m_train, idx_train, root_no, name_no)
No_test = pro.Chunk(m, False, m_train, idx_test, root_no, name_no)
n_train, n_test = No_train.data.float(), No_test.data.float()

data_n_train = get_noise_natural(data_train, n_train, snr, save_ad, sm_scale, True)
data_n_test = get_noise_natural(data_test, n_test, snr, save_ad, sm_scale, False)

test_dataset = pro.SelfData(data_n_test, sm_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
criterion = torch.nn.MSELoss().to(device)

"""
EQGraphNet
"""
EQG = net.EQGraphNet("gcn", "ts_un", 1, device).to(device)
EQG.load_state_dict(torch.load(osp.join(re_ad, "EQGraphNet", "model_{}_{}_{}_{}.pkl".
                                        format(sm_scale, name_eq, m_train, m_test))))

train_pred, train_true, test_pred, test_true = [], [], [], []
train_pos, test_pos = [], []
for item_test, (x_test, y_test, _) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)

    output_test = EQG(x_test)
    test_pred_one = output_test.detach().cpu().numpy()
    test_true_one = y_test.detach().cpu().numpy()
    if item_test == 0:
        test_pred = test_pred_one
        test_true = test_true_one
    else:
        test_pred = np.concatenate((test_pred, test_pred_one), axis=0)
        test_true = np.concatenate((test_true, test_true_one), axis=0)

rmse_EQG = net.cal_rmse_one_arr(test_true, test_pred)
r2_EQG = net.cal_r2_one_arr(test_true, test_pred)
print("EQG: RMSE_Test: {:.4f}  R2_Test: {:.4f}".format(rmse_EQG, r2_EQG))

"""
MagNet
"""
MagNet = net.MagNet().to(device)
MagNet.load_state_dict(torch.load(osp.join(re_ad, "MagNet", "model_{}_{}_{}_{}.pkl".
                                           format(sm_scale, name_eq, m_train, m_test))))

train_pred, train_true, test_pred, test_true = [], [], [], []
for item_test, (x_test, y_test, _) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)

    output_test = MagNet(x_test)
    test_pred_one = output_test.detach().cpu().numpy()
    test_true_one = y_test.detach().cpu().numpy()
    if item_test == 0:
        test_pred = test_pred_one
        test_true = test_true_one
    else:
        test_pred = np.concatenate((test_pred, test_pred_one), axis=0)
        test_true = np.concatenate((test_true, test_true_one), axis=0)

rmse_Mag = net.cal_rmse_one_arr(test_true, test_pred)
r2_Mag = net.cal_r2_one_arr(test_true, test_pred)
print("Mag: RMSE_Test: {:.4f}  R2_Test: {:.4f}".format(rmse_Mag, r2_Mag))

"""
ConvNetQuake_INGV
"""
COI = net.ConvNetQuakeINGV().to(device)
COI.load_state_dict(torch.load(osp.join(re_ad, "ConvNetQuake_INGV", "model_{}_{}_{}_{}.pkl".
                                        format(sm_scale, name_eq, m_train, m_test))))

train_pred, train_true, test_pred, test_true = [], [], [], []
for item_test, (x_test, y_test, _) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)

    output_test = COI(x_test)
    test_pred_one = output_test.detach().cpu().numpy()
    test_true_one = y_test.detach().cpu().numpy()
    if item_test == 0:
        test_pred = test_pred_one
        test_true = test_true_one
    else:
        test_pred = np.concatenate((test_pred, test_pred_one), axis=0)
        test_true = np.concatenate((test_true, test_true_one), axis=0)

rmse_COI = net.cal_rmse_one_arr(test_true, test_pred)
r2_COI = net.cal_r2_one_arr(test_true, test_pred)
print("COI: RMSE_Test: {:.4f}  R2_Test: {:.4f}".format(rmse_COI, r2_COI))

"""
CREIME
"""
def cal_mag(output):
    output_last = output[:, -10:]
    mag = torch.mean(output_last, dim=1)
    return mag


def get_xy(data, df, sm, p_len):
    data, sm = data.numpy(), sm.numpy()
    num = data.shape[0]
    p_as = df["p_arrival_sample"].values.reshape(-1).astype(int)
    n_len = 512 - p_len
    y_n_i = np.ones(shape=(1, n_len)) * (-4)
    x, y = np.zeros(shape=(num, 3, 512)), np.zeros(shape=(num, 512))
    for i in range(num):
        p_as_i, sm_i = p_as[i], sm[i]
        if p_as_i > n_len:
            x_i = data[i, :, (p_as_i - n_len): (p_as_i + p_len)]
            y_i = np.hstack([y_n_i, np.ones(shape=(1, p_len)) * sm_i])
        else:
            x_i = data[i, :, :512]
            y_i = np.hstack([np.ones(shape=(1, p_as_i)) * (-4), np.ones(shape=(1, 512 - p_as_i)) * sm_i])

        x[i, :, :] = x_i
        y[i, :] = y_i
    x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    return x, y


p_len = 125
data_n_cre_test, _ = get_xy(data_n_test, df_test, sm_test, p_len)
test_dataset = pro.SelfData(data_n_cre_test, sm_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

CREIME = net.CREIME().to(device)
CREIME.load_state_dict(torch.load(osp.join(re_ad, "CREIME", "model_{}_{}_{}_{}.pkl".
                                           format(sm_scale, name_eq, m_train, m_test))))

train_pred, train_true, test_pred, test_true = [], [], [], []
for item_test, (x_test, y_test, _) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)

    output_test = CREIME(x_test)
    test_pred_one = cal_mag(output_test).detach().cpu().numpy()
    test_true_one = y_test.detach().cpu().numpy()
    if item_test == 0:
        test_pred = test_pred_one
        test_true = test_true_one
    else:
        test_pred = np.concatenate((test_pred, test_pred_one), axis=0)
        test_true = np.concatenate((test_true, test_true_one), axis=0)

rmse_CRE = net.cal_rmse_one_arr(test_true, test_pred)
r2_CRE = net.cal_r2_one_arr(test_true, test_pred)
print("CRE: RMSE_Test: {:.4f}  R2_Test: {:.4f}".format(rmse_CRE, r2_CRE))

"""
MagInfoNet
"""
# Get P/S wave Arrival Time
ps_at_name = ["p_arrival_sample", "s_arrival_sample"]
ps_at_train, ps_at_test = df_train.loc[:, ps_at_name].values, df_test.loc[:, ps_at_name].values
_, _, ps_at_test = pro.prep_pt("sta", ps_at_train, ps_at_test)
ps_at_test = torch.from_numpy(ps_at_test).float()

# Get P wave Travel Time
t_name = ["p_travel_sec"]
p_t_train, p_t_test = df_train.loc[:, t_name].values, df_test.loc[:, t_name].values
_, _, p_t_test = pro.prep_pt("sta", p_t_train, p_t_test)
p_t_test = torch.from_numpy(p_t_test).float()

MaI = net.MagInfoNet("unimp", "ts_un", 2, device).to(device)
MaI.load_state_dict(torch.load(osp.join(re_ad, "MagInf", "model_{}_{}_{}_{}.pkl".format(sm_scale, name_eq, m_train, m_test))))

test_dataset = pro.SelfData(data_n_test, ps_at_test, p_t_test, sm_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

test_pred, test_true = [], []
for item_test, (x_test, ps_at_test, p_t_test, y_test, _) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)
    ps_at_test, p_t_test = ps_at_test.to(device), p_t_test.to(device)

    output_test = MaI(x_test, ps_at_test, p_t_test)
    test_pred_one = output_test.detach().cpu().numpy()
    test_true_one = y_test.detach().cpu().numpy()
    if item_test == 0:
        test_pred = test_pred_one
        test_true = test_true_one
    else:
        test_pred = np.concatenate((test_pred, test_pred_one), axis=0)
        test_true = np.concatenate((test_true, test_true_one), axis=0)

rmse_MaI = net.cal_rmse_one_arr(test_true, test_pred)
r2_MaI = net.cal_r2_one_arr(test_true, test_pred)
print("MaI: RMSE_Test: {:.4f}  R2_Test: {:.4f}".format(rmse_MaI, r2_MaI))

"""
save robust result
"""
if save_txt:
    info_txt_ad = osp.join(save_ad, "robust_mag_result_{}_{}_{}.txt".format(sm_scale, name_eq, m))
    info_df_ad = osp.join(save_ad, "robust_mag_result_{}_{}_{}.csv".format(sm_scale, name_eq, m))
    f = open(info_txt_ad, 'a')
    if osp.getsize(info_txt_ad) == 0:
        f.write("snr r2_MaI r2_EQG r2_Mag r2_CRE r2_COI rmse_MaI rmse_EQG rmse_Mag rmse_CRE rmse_COI\n")
    f.write(str(snr) + "  ")
    f.write(str(round(r2_MaI, 4)) + "  ")
    f.write(str(round(r2_EQG, 4)) + "  ")
    f.write(str(round(r2_Mag, 4)) + "  ")
    f.write(str(round(r2_CRE, 4)) + "  ")
    f.write(str(round(r2_COI, 4)) + "  ")
    f.write(str(round(rmse_MaI, 4)) + "  ")
    f.write(str(round(rmse_EQG, 4)) + "  ")
    f.write(str(round(rmse_Mag, 4)) + "  ")
    f.write(str(round(rmse_CRE, 4)) + "  ")
    f.write(str(round(rmse_COI, 4)) + "  ")

    f.write("\n")
    f.close()

    info = np.loadtxt(info_txt_ad, dtype=str)
    columns = info[0, :]
    values = info[1:, :]
    values_sort = sort_values(values)
    info = np.vstack([columns.reshape(1, -1), values_sort])
    np.savetxt(info_txt_ad, info, fmt='%s')
    info_df = pd.DataFrame(values_sort, columns=columns)
    info_df.to_csv(info_df_ad)

# idx = 1
# plt.figure()
# plt.plot(xn_train.numpy()[idx, 0, :], label="Noisy", alpha=0.5)
# plt.plot(n_train.numpy()[idx, 0, :], label="Noise", alpha=0.5)
# plt.plot(x_train.numpy()[idx, 0, :], label="Signal", alpha=0.5)
# plt.legend()

print()
plt.show()
print()
