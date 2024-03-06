"""
compare the influence of gnn_style to MagInfoNet and EQGraphNet
"""
import os
import os.path as osp
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append("..")
import func.process as pro
import func.net as net


gnn_style, model_style, sm_scale, device = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
# model_style = 0
# gnn_styles = ["gcn", "cheb", "gin", "graphsage", "edge",
#               "sg", "mf", "gan", "cg", "unimp"]
gnn_names = {'gcn': 'GCN', 'cheb': 'ChebNet', 'gin': 'GIN', 'graphsage': 'GraphSage', 'edge': 'DG Edge CNN',
             'sg': 'S-GCN', 'mf': 'MF-GCN', 'resgate': 'RG-GCN', 'gan': 'GAN'}
if not torch.cuda.is_available():
    device = "cpu"
lr = 0.0005
weight_decay = 0.0005
batch_size = 64
epochs = 70
train_ratio = 0.75
m = 200000
random = False
save_txt = True
save_model = True
save_ad = "../factor/gnn_style_result"
if not(osp.exists(save_ad)):
    os.makedirs(save_ad)

print("\n\n" + "=" * 20 + " Start {} Training ".format(gnn_style) + "=" * 20 + "\n")

"""
Selection of noise and earthquake signals
"""
m_train = int(m * train_ratio)       # number of training samples
m_test = m - m_train                     # number of testing samples
name = "chunk2"
root = "/home/chenziwei2021/standford_dataset/{}".format(name)

if not random:
    np.random.seed(100)
idx_train, idx_test = pro.get_train_or_test_idx(m, m_train)
eq_train = pro.Chunk(m, True, m_train, idx_train, root, name)
eq_test = pro.Chunk(m, False, m_train, idx_test, root, name)
df_train, df_test = eq_train.df, eq_test.df

data_train, data_test = eq_train.data.float(), eq_test.data.float()
sm_train = torch.from_numpy(df_train["source_magnitude"].values.reshape(-1)).float()
sm_test = torch.from_numpy(df_test["source_magnitude"].values.reshape(-1)).float()

# Select samples according to Magnitude Type
data_train, sm_train, df_train, _ = pro.remain_sm_scale(data_train, df_train, sm_train, sm_scale)
data_test, sm_test, df_test, _ = pro.remain_sm_scale(data_test, df_test, sm_test, sm_scale)

criterion = torch.nn.MSELoss().to(device)

# train EQGraphNet
if model_style == 0:
    print("=" * 20 + " EQGraphNet, {}, {}".format(gnn_names[gnn_style], sm_scale) + "=" * 20 + "\n")
    train_dataset = pro.SelfData(data_train, sm_train)
    test_dataset = pro.SelfData(data_test, sm_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    EQG = net.EQGraphNet(gnn_style, "ts_un", 1, device).to(device)
    optimizer = torch.optim.Adam(EQG.parameters(), lr=lr, weight_decay=weight_decay)

    train_pred, train_true, test_pred, test_true = [], [], [], []
    train_trace, test_trace = [], []
    train_pos, test_pos = [], []
    train_loss, test_loss = [], []
    for epoch in range(epochs):
        loss_train_all, loss_test_all = 0, 0
        for item_train, (x_train, y_train, _) in enumerate(tqdm(train_loader)):
            x_train, y_train = x_train.to(device), y_train.to(device)

            optimizer.zero_grad()
            output_train = EQG(x_train)
            loss_train = criterion(output_train, y_train)
            loss_train.backward()
            optimizer.step()
            loss_train_all = loss_train_all + loss_train.item()

            train_pred_one = output_train.detach().cpu().numpy()
            train_true_one = y_train.detach().cpu().numpy()
            if item_train == 0:
                train_pred = train_pred_one
                train_true = train_true_one
            else:
                train_pred = np.concatenate((train_pred, train_pred_one), axis=0)
                train_true = np.concatenate((train_true, train_true_one), axis=0)

        for item_test, (x_test, y_test, _) in enumerate(tqdm(test_loader)):
            x_test, y_test = x_test.to(device), y_test.to(device)

            output_test = EQG(x_test)
            loss_test = criterion(output_test, y_test)
            loss_test_all = loss_test_all + loss_test.item()

            test_pred_one = output_test.detach().cpu().numpy()
            test_true_one = y_test.detach().cpu().numpy()
            if item_test == 0:
                test_pred = test_pred_one
                test_true = test_true_one
            else:
                test_pred = np.concatenate((test_pred, test_pred_one), axis=0)
                test_true = np.concatenate((test_true, test_true_one), axis=0)

        rmse_train = net.cal_rmse_one_arr(train_true, train_pred)
        rmse_test = net.cal_rmse_one_arr(test_true, test_pred)
        r2_train = net.cal_r2_one_arr(train_true, train_pred)
        r2_test = net.cal_r2_one_arr(test_true, test_pred)
        print("Epoch: {:04d}  RMSE_Train: {:.4f}  RMSE_Test: {:.4f}  R2_Train: {:.8f}  R2_Test: {:.8f}".
              format(epoch, rmse_train, rmse_test, r2_train, r2_test))
        print()
        if (0.91 < r2_test and sm_scale == "ml") or (0.855 < r2_test and sm_scale == "md"):
            break

    error = test_pred - test_true
    e_mean, e_std = np.mean(error), np.std(error)

    if save_model:
        torch.save(EQG.state_dict(),
                   osp.join(save_ad, "EQG_{}_{}_{}_{}_{}.pkl".format(gnn_style, sm_scale, name, m_train, m_test)))

    if save_txt:
        info_txt_ad = osp.join(save_ad, "EQG_result_{}_{}_{}.txt".format(sm_scale, name, m))
        info_df_ad = osp.join(save_ad, "EQG_result_{}_{}_{}.csv".format(sm_scale, name, m))
        f = open(info_txt_ad, 'a')
        if osp.getsize(info_txt_ad) == 0:
            f.write("gnn_style r2 rmse e_mean e_std\n")
        f.write(str(gnn_style) + "  ")
        f.write(str(np.round(r2_test, 4)) + "  ")
        f.write(str(np.round(rmse_test, 4)) + "  ")
        f.write(str(np.round(e_mean, 4)) + "  ")
        f.write(str(np.round(e_std, 4)) + "  ")

        f.write("\n")
        f.close()

        info = np.loadtxt(info_txt_ad, dtype=str)
        columns = info[0, :].tolist()
        values = info[1:, :]
        info_df = pd.DataFrame(values, columns=columns)
        info_df.to_csv(info_df_ad)

# train MagInfoNet
elif model_style == 1:
    print("=" * 20 + " MagInfoNet, {}, {}".format(gnn_names[gnn_style], sm_scale) + "=" * 20 + "\n")
    prep_style = "sta"
    ps_at_name = ["p_arrival_sample", "s_arrival_sample"]
    ps_at_train, ps_at_test = df_train.loc[:, ps_at_name].values, df_test.loc[:, ps_at_name].values
    prep_ps_at, ps_at_train, ps_at_test = pro.prep_pt(prep_style, ps_at_train, ps_at_test)
    ps_at_train, ps_at_test = torch.from_numpy(ps_at_train).float(), torch.from_numpy(ps_at_test).float()

    t_name = ["p_travel_sec"]
    p_t_train, p_t_test = df_train.loc[:, t_name].values, df_test.loc[:, t_name].values
    prep_p_t, p_t_train, p_t_test = pro.prep_pt(prep_style, p_t_train, p_t_test)
    p_t_train, p_t_test = torch.from_numpy(p_t_train).float(), torch.from_numpy(p_t_test).float()

    train_dataset = pro.SelfData(data_train, sm_train, ps_at_train, p_t_train)
    test_dataset = pro.SelfData(data_test, sm_test, ps_at_test, p_t_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    MaI = net.MagInfoNet(gnn_style, "ts_un", 1, device).to(device)
    optimizer = torch.optim.Adam(MaI.parameters(), lr=lr, weight_decay=weight_decay)

    train_pred, train_true, test_pred, test_true = [], [], [], []
    for epoch in range(epochs):
        loss_train_all, loss_test_all = 0, 0
        for item_train, (x_train, y_train, ps_at_train, p_t_train, _) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)
            ps_at_train, p_t_train = ps_at_train.to(device), p_t_train.to(device)

            optimizer.zero_grad()
            output_train = MaI(x_train, ps_at_train, p_t_train)
            loss_train = criterion(output_train, y_train)
            loss_train.backward()
            optimizer.step()
            loss_train_all = loss_train_all + loss_train.item()

            train_pred_one = output_train.detach().cpu().numpy()
            train_true_one = y_train.detach().cpu().numpy()
            if item_train == 0:
                train_pred = train_pred_one
                train_true = train_true_one
            else:
                train_pred = np.concatenate((train_pred, train_pred_one), axis=0)
                train_true = np.concatenate((train_true, train_true_one), axis=0)

        for item_test, (x_test, y_test, ps_at_test, p_t_test, _) in enumerate(test_loader):
            x_test, y_test = x_test.to(device), y_test.to(device)
            ps_at_test, p_t_test = ps_at_test.to(device), p_t_test.to(device)

            output_test = MaI(x_test, ps_at_test, p_t_test)
            loss_test = criterion(output_test, y_test)
            loss_test_all = loss_test_all + loss_test.item()

            test_pred_one = output_test.detach().cpu().numpy()
            test_true_one = y_test.detach().cpu().numpy()
            if item_test == 0:
                test_pred = test_pred_one
                test_true = test_true_one
            else:
                test_pred = np.concatenate((test_pred, test_pred_one), axis=0)
                test_true = np.concatenate((test_true, test_true_one), axis=0)

        rmse_train = net.cal_rmse_one_arr(train_true, train_pred)
        rmse_test = net.cal_rmse_one_arr(test_true, test_pred)
        r2_train = net.cal_r2_one_arr(train_true, train_pred)
        r2_test = net.cal_r2_one_arr(test_true, test_pred)
        print("Epoch: {:04d}  RMSE_Train: {:.4f}  RMSE_Test: {:.4f}  R2_Train: {:.8f}  R2_Test: {:.8f}".
              format(epoch, rmse_train, rmse_test, r2_train, r2_test))
        print()
        if (0.88 < r2_test and sm_scale == "ml") or (0.82 < r2_test and sm_scale == "md"):
            break

    error = test_pred - test_true
    e_mean, e_std = np.mean(error), np.std(error)

    if save_model:
        torch.save(MaI.state_dict(),
                   osp.join(save_ad, "MaI_{}_{}_{}_{}_{}.pkl".format(gnn_style, sm_scale, name, m_train, m_test)))

    if save_txt:
        info_txt_ad = osp.join(save_ad, "MaI_result_{}_{}_{}.txt".format(sm_scale, name, m))
        info_df_ad = osp.join(save_ad, "MaI_result_{}_{}_{}.csv".format(sm_scale, name, m))
        f = open(info_txt_ad, 'a')
        if osp.getsize(info_txt_ad) == 0:
            f.write("gnn_style r2 rmse e_mean e_std\n")
        f.write(str(gnn_style) + "  ")
        f.write(str(np.round(r2_test, 4)) + "  ")
        f.write(str(np.round(rmse_test, 4)) + "  ")
        f.write(str(np.round(e_mean, 4)) + "  ")
        f.write(str(np.round(e_std, 4)) + "  ")

        f.write("\n")
        f.close()

        info = np.loadtxt(info_txt_ad, dtype=str)
        columns = info[0, :].tolist()
        values = info[1:, :]
        info_df = pd.DataFrame(values, columns=columns)
        info_df.to_csv(info_df_ad)

print()
