"""
Functions for analysing the results
"""
import numpy as np
import os.path as osp
import pandas as pd


def cal_rmse_one_arr(true, pred):
    return np.sqrt(np.mean(np.square(true - pred)))


def cal_r2_one_arr(true, pred):
    corr_matrix = np.corrcoef(true, pred)
    corr = corr_matrix[0, 1]
    r2 = corr ** 2
    return r2


def judge_idx(*args):
    def get_trace_idx(trace, trace_i):
        idx_ = np.argwhere(trace == trace_i).reshape(-1)
        if idx_.shape[0] == 0:
            raise ValueError("!")
        else:
            return idx_[0]

    num = args[0].shape[0]          # number of samples in this trace
    idx = [[] for _ in range(len(args))]
    for i in range(num):
        for j in range(len(args)):
            if j == 0:
                idx[j].append(i)
            else:
                idx[j].append(get_trace_idx(args[j], args[0][i]))
    return tuple(idx)


def tran(pred, true, pos, trace, idx):
    pred_ = pred[idx]
    true_ = true[idx]
    pos_ = pos[idx, :]
    trace_ = trace[idx]
    return pred_, true_, pos_, trace_


# remain samples in given range
def select_range(pos, true, trace, lat_min, lat_max, lon_min, lon_max, *args):
    lon, lat = pos[:, 0], pos[:, 1]
    idx = np.argwhere((lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)).reshape(-1)
    if idx.shape[0] == 0:
        raise ValueError("The ranges of lat and lon is too small! Change or increase it.")
    pos_, true_, trace_ = pos[idx, :], true[idx], trace[idx]
    res = [pos_, true_, trace_]
    for arg in args:
        arg_ = arg[idx]
        res.append(arg_)
    return tuple(res)


# read npy outputted by Networks
def read_npy(ad, sm_i, name, m_train, m_test):
    pred = np.load(osp.join(ad, "test_pred_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
    true = np.load(osp.join(ad, "test_true_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
    pos = np.load(osp.join(ad, "test_pos_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
    trace = np.load(osp.join(ad, "test_trace_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
    return pred, true, pos, trace


# read estimated results of magnitude
def read_sm(sm_list, re_ad, name, m_train, m_test):
    m = m_train + m_test
    pos_ad = osp.join(re_ad, "pos_{}_{}.npy".format(name, m))
    true_ad = osp.join(re_ad, "true_{}_{}.npy".format(name, m))
    trace_ad = osp.join(re_ad, "trace_{}_{}.npy".format(name, m))
    pred_MaI_ad = osp.join(re_ad, "pred_MaI_{}_{}.npy".format(name, m))
    pred_EQG_ad = osp.join(re_ad, "pred_EQG_{}_{}.npy".format(name, m))
    pred_Mag_ad = osp.join(re_ad, "pred_Mag_{}_{}.npy".format(name, m))
    pred_CRE_ad = osp.join(re_ad, "pred_CRE_{}_{}.npy".format(name, m))
    pred_COI_ad = osp.join(re_ad, "pred_COI_{}_{}.npy".format(name, m))

    if osp.exists(pos_ad) & osp.exists(true_ad) & osp.exists(pred_EQG_ad) & osp.exists(pred_Mag_ad) & \
            osp.exists(pred_CRE_ad) & osp.exists(pred_COI_ad) & osp.exists(trace_ad):
        pos_ = np.load(pos_ad)
        true_ = np.load(true_ad)
        trace_ = np.load(trace_ad)
        pred_MaI_ = np.load(pred_MaI_ad)
        pred_EQG_ = np.load(pred_EQG_ad)
        pred_Mag_ = np.load(pred_Mag_ad)
        pred_CRE_ = np.load(pred_CRE_ad)
        pred_COI_ = np.load(pred_COI_ad)
    else:
        num = len(sm_list)
        pos_, true_, trace_, pred_MaI_, pred_EQG_, pred_Mag_, pred_CRE_, pred_COI_ = 0, 0, 0, 0, 0, 0, 0, 0
        for i in range(num):
            sm_i = sm_list[i]

            pred_MaI, true_MaI, pos_MaI, trace_MaI = read_npy(osp.join(re_ad, "MagInf"), sm_i, name, m_train, m_test)
            pred_EQG, true_EQG, pos_EQG, trace_EQG = read_npy(osp.join(re_ad, "EQGraphNet"), sm_i, name, m_train, m_test)
            pred_Mag, true_Mag, pos_Mag, trace_Mag = read_npy(osp.join(re_ad, "MagNet"), sm_i, name, m_train, m_test)
            pred_CRE, true_CRE, pos_CRE, trace_CRE = read_npy(osp.join(re_ad, "CREIME"), sm_i, name, m_train, m_test)
            pred_COI, true_COI, pos_COI, trace_COI = read_npy(osp.join(re_ad, "ConvNetQuake_INGV"), sm_i, name, m_train, m_test)

            idx_MaI, idx_EQG, idx_Mag, idx_CRE, idx_COI = judge_idx(trace_MaI, trace_EQG, trace_Mag, trace_CRE, trace_COI)
            pred_MaI, true_MaI, pos_MaI, trace_MaI = tran(pred_MaI, true_MaI, pos_MaI, trace_MaI, idx_MaI)
            pred_EQG, true_EQG, pos_EQG, trace_EQG = tran(pred_EQG, true_EQG, pos_EQG, trace_EQG, idx_EQG)
            pred_Mag, true_Mag, pos_Mag, trace_Mag = tran(pred_Mag, true_Mag, pos_Mag, trace_Mag, idx_Mag)
            pred_CRE, true_CRE, pos_CRE, trace_CRE = tran(pred_CRE, true_CRE, pos_CRE, trace_CRE, idx_CRE)
            pred_COI, true_COI, pos_COI, trace_COI = tran(pred_COI, true_COI, pos_COI, trace_COI, idx_COI)
            pos, true, trace = pos_Mag, true_Mag, trace_Mag       # all models are the same

            if i == 0:
                pos_, true_, trace_ = pos, true, trace
                pred_MaI_, pred_EQG_, pred_Mag_, pred_CRE_, pred_COI_ = pred_MaI, pred_EQG, pred_Mag, pred_CRE, pred_COI
            else:
                pos_ = np.concatenate((pos_, pos), axis=0)
                true_ = np.concatenate((true_, true), axis=0)
                trace_ = np.concatenate((trace_, trace), axis=0)
                pred_MaI_ = np.concatenate((pred_MaI_, pred_MaI), axis=0)
                pred_EQG_ = np.concatenate((pred_EQG_, pred_EQG), axis=0)
                pred_Mag_ = np.concatenate((pred_Mag_, pred_Mag), axis=0)
                pred_CRE_ = np.concatenate((pred_CRE_, pred_CRE), axis=0)
                pred_COI_ = np.concatenate((pred_COI_, pred_COI), axis=0)

        np.save(pos_ad, pos_)
        np.save(true_ad, true_)
        np.save(trace_ad, trace_)
        np.save(pred_MaI_ad, pred_MaI_)
        np.save(pred_EQG_ad, pred_EQG_)
        np.save(pred_Mag_ad, pred_Mag_)
        np.save(pred_CRE_ad, pred_CRE_)
        np.save(pred_COI_ad, pred_COI_)

    return pos_, true_, trace_, pred_MaI_, pred_EQG_, pred_Mag_, pred_CRE_, pred_COI_


def select_trace(trace, df, re_ad, name, m):
    df_idx_ad = osp.join(re_ad, "df_idx_{}_{}.npy".format(name, m))
    if osp.exists(df_idx_ad):
        idx = np.load(df_idx_ad)
    else:
        df_trace = df["trace_name"].values.reshape(-1)
        num = trace.shape[0]
        idx = []
        for i in range(num):
            trace_i = trace[i]
            idx_i = np.argwhere(df_trace == trace_i).reshape(-1)
            if idx_i.shape[0] == 0:
                raise ValueError("!")
            else:
                idx.append(idx_i[0])
        idx = np.array(idx)
        np.save(df_idx_ad, idx)
    df_ = df.iloc[idx, :]
    return df_


def select_trace_small(trace, df):
    df_trace = df["trace_name"].values.reshape(-1)
    num = trace.shape[0]
    idx = []
    for i in range(num):
        trace_i = trace[i]
        idx_i = np.argwhere(df_trace == trace_i).reshape(-1)
        if idx_i.shape[0] == 0:
            raise ValueError("!")
        else:
            idx.append(idx_i[0])
    idx = np.array(idx)
    df_ = df.iloc[idx, :]
    return df_


def load_txt(file_ad):
    info = np.loadtxt(file_ad, dtype=str)
    columns, values = info[0, :], info[1:, :]
    df = pd.DataFrame(values, columns=columns)
    return df


def get_eq_info(root, name, trace):
    df = pd.read_csv(osp.join(root, name + ".csv"))
    df_trace = df.loc[:, "trace_name"].values.reshape(-1)
    idx = []
    for trace_one in trace:
        idx_one = np.argwhere(df_trace == trace_one).reshape(-1)[0]
        idx.append(idx_one)
    idx = np.sort(np.array(idx))
    df = df.iloc[idx, :]
    return df
