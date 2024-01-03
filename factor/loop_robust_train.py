"""
Natural anti-Noise performance, we train and test the model
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
    if e_n < 0.1:                               # 有些噪声全为零值
        n = np.random.normal(size=x.shape)
        e_n = np.mean(np.square(n))
    ratio = np.sqrt(e_x / e_n / np.power(10, (snr / 10)))
    if ratio > 10000:
        b = 1
    n_ = n * ratio
    n_ = np.expand_dims(n_, axis=0)
    snr_true = 10 * np.log10(np.mean(np.square(x)) / np.mean(np.square(n_)))
    return n_


def get_noise_natural(x, n, snr):
    x, n = x.numpy(), n.numpy()
    xn = copy.deepcopy(x)
    num_sample = x.shape[0]
    for i in range(num_sample):
        x_one, n_one = x[i, :, :], n[i, :, :]
        x_one_1, x_one_2, x_one_3 = x_one[0, :], x_one[1, :], x_one[2, :]
        n_one_1, n_one_2, n_one_3 = n_one[0, :], n_one[1, :], n_one[2, :]
        n_one_1 = get_noise_one_natural(x_one_1, n_one_1, snr)
        n_one_2 = get_noise_one_natural(x_one_2, n_one_2, snr)
        n_one_3 = get_noise_one_natural(x_one_3, n_one_3, snr)
        n_one = np.concatenate((n_one_1, n_one_2, n_one_3), axis=0)
        n_one = np.expand_dims(n_one, axis=0)
        x_n_one = x_one + n_one
        xn[i, :, :] = xn[i, :, :] + x_n_one
    xn, n = torch.from_numpy(xn).float(), torch.from_numpy(n).float()
    return xn, n


snr_list = [3, 4, 5, 10, 15]

for snr in snr_list:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    train_ratio = 0.75
    m = 200000
    epochs = 70
    sm_scale = "md"
    random = False
    save_txt = True
    save_ad = "../factor/robust_train_result"
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
    idx_train_eq, idx_test_eq = pro.get_train_or_test_idx(m, m_train)

    Eq_train = pro.Chunk(m, True, m_train, idx_train_eq, root_eq, name_eq)
    Eq_test = pro.Chunk(m, False, m_train, idx_test_eq, root_eq, name_eq)
    df_train, df_test = Eq_train.df, Eq_test.df
    x_train, x_test = Eq_train.data.float(), Eq_test.data.float()
    sm_train = torch.from_numpy(df_train["source_magnitude"].values.reshape(-1)).float()
    sm_test = torch.from_numpy(df_test["source_magnitude"].values.reshape(-1)).float()

    # Select samples according to Magnitude Type
    x_train, sm_train, df_train, _ = pro.remain_sm_scale(x_train, df_train, sm_train, sm_scale)
    x_test, sm_test, df_test, _ = pro.remain_sm_scale(x_test, df_test, sm_test, sm_scale)

    # get natural noise
    m_train_no, m_test_no = x_train.shape[0], x_test.shape[0]
    m_no = m_train_no + m_test_no
    idx_train_no, idx_test_no = pro.get_train_or_test_idx(m_no, m_train_no)
    No_train = pro.Chunk(m_no, True, m_train_no, idx_train_no, root_no, name_no)
    No_test = pro.Chunk(m_no, False, m_train_no, idx_test_no, root_no, name_no)
    n_train, n_test = No_train.data.float(), No_test.data.float()

    xn_train, n_train = get_noise_natural(x_train, n_train, snr)
    xn_test, n_test = get_noise_natural(x_test, n_test, snr)

    train_dataset = pro.SelfData(xn_train, sm_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = pro.SelfData(xn_test, sm_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.MSELoss().to(device)

    """
    EQGraphNet
    """
    print("\nEQGraphNet, training {}".format("-" * 30))
    EQG = net.EQGraphNet("gcn", "ts_un", 1, device).to(device)
    optimizer = torch.optim.Adam(EQG.parameters(), lr=0.0005, weight_decay=0.0005)

    train_pred, train_true, test_pred, test_true = [], [], [], []
    for epoch in range(epochs):
        loss_train_all, loss_test_all = 0, 0
        for item_train, (x_train, y_train, _) in enumerate(train_loader):
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

        for item_test, (x_test, y_test, _) in enumerate(test_loader):
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

        rmse_train_EQG = net.cal_rmse_one_arr(train_true, train_pred)
        rmse_test_EQG = net.cal_rmse_one_arr(test_true, test_pred)
        r2_train_EQG = net.cal_r2_one_arr(train_true, train_pred)
        r2_test_EQG = net.cal_r2_one_arr(test_true, test_pred)
        print("Epoch: {:04d}  RMSE_Train: {:.4f}  RMSE_Test: {:.4f}  R2_Train: {:.8f}  R2_Test: {:.8f}".
              format(epoch, rmse_train_EQG, rmse_test_EQG, r2_train_EQG, r2_test_EQG))

    """
    MagNet
    """
    print("\nMagNet, training {}".format("-" * 30))
    Mag = net.MagNet().to(device)
    optimizer = torch.optim.Adam(Mag.parameters(), lr=0.0005, weight_decay=0.0005)

    train_pred, train_true, test_pred, test_true = [], [], [], []
    for epoch in range(epochs):
        loss_train_all, loss_test_all = 0, 0
        for item_train, (x_train, y_train, _) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)

            optimizer.zero_grad()
            output_train = Mag(x_train)
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

        for item_test, (x_test, y_test, _) in enumerate(test_loader):
            x_test, y_test = x_test.to(device), y_test.to(device)

            output_test = Mag(x_test)
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

        rmse_train_Mag = net.cal_rmse_one_arr(train_true, train_pred)
        rmse_test_Mag = net.cal_rmse_one_arr(test_true, test_pred)
        r2_train_Mag = net.cal_r2_one_arr(train_true, train_pred)
        r2_test_Mag = net.cal_r2_one_arr(test_true, test_pred)
        print("Epoch: {:04d}  RMSE_Train: {:.4f}  RMSE_Test: {:.4f}  R2_Train: {:.8f}  R2_Test: {:.8f}".
              format(epoch, rmse_train_Mag, rmse_test_Mag, r2_train_Mag, r2_test_Mag))

    """
    CNQI
    """
    print("\nConvNetQuake_INGV, training {}".format("-" * 30))
    CNQI = net.ConvNetQuakeINGV().to(device)
    optimizer = torch.optim.Adam(CNQI.parameters(), lr=0.0005, weight_decay=0.0005)

    train_pred, train_true, test_pred, test_true = [], [], [], []
    for epoch in range(epochs):
        loss_train_all, loss_test_all = 0, 0
        for item_train, (x_train, y_train, _) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)

            optimizer.zero_grad()
            output_train = CNQI(x_train)
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

        for item_test, (x_test, y_test, _) in enumerate(test_loader):
            x_test, y_test = x_test.to(device), y_test.to(device)

            output_test = CNQI(x_test)
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

        rmse_train_CNQI = net.cal_rmse_one_arr(train_true, train_pred)
        rmse_test_CNQI = net.cal_rmse_one_arr(test_true, test_pred)
        r2_train_CNQI = net.cal_r2_one_arr(train_true, train_pred)
        r2_test_CNQI = net.cal_r2_one_arr(test_true, test_pred)
        print("Epoch: {:04d}  RMSE_Train: {:.4f}  RMSE_Test: {:.4f}  R2_Train: {:.8f}  R2_Test: {:.8f}".
              format(epoch, rmse_train_CNQI, rmse_test_CNQI, r2_train_CNQI, r2_test_CNQI))

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
    x_train, y_train = get_xy(xn_train, df_train, sm_train, p_len)
    x_test, y_test = get_xy(xn_test, df_test, sm_test, p_len)
    train_dataset = pro.SelfData(x_train, y_train, sm_train)
    test_dataset = pro.SelfData(x_test, y_test, sm_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("\nCREIME, training {}".format("-" * 30))
    CRE = net.CREIME().to(device)
    optimizer = torch.optim.Adam(CRE.parameters(), lr=0.0005, weight_decay=0.0005)

    train_pred, train_true, test_pred, test_true = [], [], [], []
    for epoch in range(epochs):
        loss_train_all, loss_test_all = 0, 0
        for item_train, (x_train, y_train, sm_train, _) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)

            optimizer.zero_grad()
            output_train = CRE(x_train)
            loss_train = criterion(output_train, y_train)
            loss_train.backward()
            optimizer.step()
            loss_train_all = loss_train_all + loss_train.item()

            train_pred_one = cal_mag(output_train).detach().cpu().numpy()
            train_true_one = sm_train.detach().cpu().numpy()
            if item_train == 0:
                train_pred = train_pred_one
                train_true = train_true_one
            else:
                train_pred = np.concatenate((train_pred, train_pred_one), axis=0)
                train_true = np.concatenate((train_true, train_true_one), axis=0)

        for item_test, (x_test, y_test, sm_test, _) in enumerate(test_loader):
            x_test, y_test = x_test.to(device), y_test.to(device)

            output_test = CRE(x_test)
            loss_test = criterion(output_test, y_test)
            loss_test_all = loss_test_all + loss_test.item()

            test_pred_one = cal_mag(output_test).detach().cpu().numpy()
            test_true_one = sm_test.detach().cpu().numpy()
            if item_test == 0:
                test_pred = test_pred_one
                test_true = test_true_one
            else:
                test_pred = np.concatenate((test_pred, test_pred_one), axis=0)
                test_true = np.concatenate((test_true, test_true_one), axis=0)

        rmse_train_CRE = net.cal_rmse_one_arr(train_true, train_pred)
        rmse_test_CRE = net.cal_rmse_one_arr(test_true, test_pred)
        r2_train_CRE = net.cal_r2_one_arr(train_true, train_pred)
        r2_test_CRE = net.cal_r2_one_arr(test_true, test_pred)
        print("Epoch: {:04d}  RMSE_Train: {:.4f}  RMSE_Test: {:.4f}  R2_Train: {:.8f}  R2_Test: {:.8f}".
              format(epoch, rmse_train_CRE, rmse_test_CRE, r2_train_CRE, r2_test_CRE))

    """
    save robust result
    """
    if save_txt:
        info_txt_ad = osp.join(save_ad, "robust_train_result_{}_{}_{}.txt".format(sm_scale, name_eq, m))
        info_df_ad = osp.join(save_ad, "robust_train_result_{}_{}_{}.csv".format(sm_scale, name_eq, m))
        f = open(info_txt_ad, 'a')
        if osp.getsize(info_txt_ad) == 0:
            f.write("snr r2_EQG r2_MagNet r2_CREIME r2_CNQI rmse_EQG rmse_MagNet rmse_CREIME rmse_CNQI\n")
        f.write(str(snr) + "  ")
        f.write(str(round(r2_test_EQG, 4)) + "  ")
        f.write(str(round(r2_test_Mag, 4)) + "  ")
        f.write(str(round(r2_test_CRE, 4)) + "  ")
        f.write(str(round(r2_test_CNQI, 4)) + "  ")
        f.write(str(round(rmse_test_EQG, 4)) + "  ")
        f.write(str(round(rmse_test_Mag, 4)) + "  ")
        f.write(str(round(rmse_test_CRE, 4)) + "  ")
        f.write(str(round(rmse_test_CNQI, 4)) + "  ")

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

    print()

