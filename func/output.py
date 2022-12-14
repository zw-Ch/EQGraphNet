"""
Functions for analysing the results
"""
import numpy as np
import os.path as osp


def cal_rmse_one_arr(true, pred):
    return np.sqrt(np.mean(np.square(true - pred)))


def cal_r2_one_arr(true, pred):
    corr_matrix = np.corrcoef(true, pred)
    corr = corr_matrix[0, 1]
    r2 = corr ** 2
    return r2


def judge_idx(trace1, trace2, trace3, trace4):
    def get_trace_idx(trace, trace_i):
        idx = np.argwhere(trace == trace_i).reshape(-1)
        if idx.shape[0] == 0:
            raise ValueError("!")
        else:
            return idx[0]

    num = trace1.shape[0]
    idx1, idx2, idx3, idx4 = [], [], [], []
    for i in range(num):
        trace1_i = trace1[i]
        idx2_i = get_trace_idx(trace2, trace1_i)
        idx3_i = get_trace_idx(trace3, trace1_i)
        idx4_i = get_trace_idx(trace4, trace1_i)
        idx1.append(i), idx2.append(idx2_i), idx3.append(idx3_i), idx4.append(idx4_i)
    return idx1, idx2, idx3, idx4


def tran(pred, true, pos, trace, idx):
    pred_ = pred[idx]
    true_ = true[idx]
    pos_ = pos[idx, :]
    trace_ = trace[idx]
    return pred_, true_, pos_, trace_


def select_range(pos, true, trace, lat_min, lat_max, lon_min, lon_max, pred_EQG, pred_Mag, pred_COI, pred_CRE):
    lon, lat = pos[:, 0], pos[:, 1]
    idx = np.argwhere((lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)).reshape(-1)
    if idx.shape[0] == 0:
        raise ValueError("The ranges of lat and lon is too small! Change or increase it.")
    pos_, true_, trace_ = pos[idx, :], true[idx], trace[idx]
    pred_EQG_ = pred_EQG[idx]
    pred_Mag_ = pred_Mag[idx]
    pred_COI_ = pred_COI[idx]
    pred_CRE_ = pred_CRE[idx]
    return pos_, true_, trace_, pred_EQG_, pred_Mag_, pred_COI_, pred_CRE_


def read_sm(sm_list, re_ad, name, m_train, m_test):
    m = m_train + m_test
    pos_ad = osp.join(re_ad, "pos_{}_{}.npy".format(name, m))
    true_ad = osp.join(re_ad, "true_{}_{}.npy".format(name, m))
    trace_ad = osp.join(re_ad, "trace_{}_{}.npy".format(name, m))
    pred_EQG_ad = osp.join(re_ad, "pred_EQG_{}_{}.npy".format(name, m))
    pred_Mag_ad = osp.join(re_ad, "pred_Mag_{}_{}.npy".format(name, m))
    pred_COI_ad = osp.join(re_ad, "pred_COI_{}_{}.npy".format(name, m))
    pred_CRE_ad = osp.join(re_ad, "pred_CRE_{}_{}.npy".format(name, m))

    if osp.exists(pos_ad) & osp.exists(true_ad) & osp.exists(pred_EQG_ad) & osp.exists(pred_Mag_ad) & \
            osp.exists(pred_COI_ad) & osp.exists(pred_CRE_ad) & osp.exists(trace_ad):
        pos_ = np.load(pos_ad)
        true_ = np.load(true_ad)
        trace_ = np.load(trace_ad)
        pred_EQG_ = np.load(pred_EQG_ad)
        pred_Mag_ = np.load(pred_Mag_ad)
        pred_COI_ = np.load(pred_COI_ad)
        pred_CRE_ = np.load(pred_CRE_ad)
    else:
        num = len(sm_list)
        pos_, true_, trace_, pred_EQG_, pred_Mag_, pred_COI_, pred_CRE_ = 0, 0, 0, 0, 0, 0, 0
        for i in range(num):
            sm_i = sm_list[i]

            # EQGraphNet
            re_ad_EQG = osp.join(re_ad, "MagPre")
            pred_EQG = np.load(osp.join(re_ad_EQG, "test_pred_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
            true_EQG = np.load(osp.join(re_ad_EQG, "test_true_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
            pos_EQG = np.load(osp.join(re_ad_EQG, "test_pos_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
            trace_EQG = np.load(osp.join(re_ad_EQG, "test_trace_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))

            # MagNet
            re_ad_Mag = osp.join(re_ad, "MagNet")
            pred_Mag = np.load(osp.join(re_ad_Mag, "test_pred_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
            true_Mag = np.load(osp.join(re_ad_Mag, "test_true_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
            pos_Mag = np.load(osp.join(re_ad_Mag, "test_pos_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
            trace_Mag = np.load(osp.join(re_ad_Mag, "test_trace_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))

            # ConvNetQuake_INGV
            re_ad_COI = osp.join(re_ad, "ConvNetQuake_INGV")
            pred_COI = np.load(osp.join(re_ad_COI, "test_pred_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
            true_COI = np.load(osp.join(re_ad_COI, "test_true_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
            pos_COI = np.load(osp.join(re_ad_COI, "test_pos_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
            trace_COI = np.load(osp.join(re_ad_COI, "test_trace_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))

            # CREIME
            re_ad_CRE = osp.join(re_ad, "CREIME")
            pred_CRE = np.load(osp.join(re_ad_CRE, "test_pred_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
            true_CRE = np.load(osp.join(re_ad_CRE, "test_true_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
            pos_CRE = np.load(osp.join(re_ad_CRE, "test_pos_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))
            trace_CRE = np.load(osp.join(re_ad_CRE, "test_trace_{}_{}_{}_{}.npy".format(sm_i, name, m_train, m_test)))

            idx_EQG, idx_Mag, idx_COI, idx_CRE = judge_idx(trace_EQG, trace_Mag, trace_COI, trace_CRE)
            pred_EQG, true_EQG, pos_EQG, trace_EQG = tran(pred_EQG, true_EQG, pos_EQG, trace_EQG, idx_EQG)
            pred_Mag, true_Mag, pos_Mag, trace_Mag = tran(pred_Mag, true_Mag, pos_Mag, trace_Mag, idx_Mag)
            pred_COI, true_COI, pos_COI, trace_COI = tran(pred_COI, true_COI, pos_COI, trace_COI, idx_COI)
            pred_CRE, true_CRE, pos_CRE, trace_CRE = tran(pred_CRE, true_CRE, pos_CRE, trace_CRE, idx_CRE)
            pos, true, trace = pos_Mag, true_Mag, trace_Mag       # all models are the same

            if i == 0:
                pos_, true_, trace_ = pos, true, trace
                pred_EQG_, pred_Mag_, pred_COI_, pred_CRE_ = pred_EQG, pred_Mag, pred_COI, pred_CRE
            else:
                pos_ = np.concatenate((pos_, pos), axis=0)
                true_ = np.concatenate((true_, true), axis=0)
                trace_ = np.concatenate((trace_, trace), axis=0)
                pred_EQG_ = np.concatenate((pred_EQG_, pred_EQG), axis=0)
                pred_Mag_ = np.concatenate((pred_Mag_, pred_Mag), axis=0)
                pred_COI_ = np.concatenate((pred_COI_, pred_COI), axis=0)
                pred_CRE_ = np.concatenate((pred_CRE_, pred_CRE), axis=0)

        np.save(pos_ad, pos_)
        np.save(true_ad, true_)
        np.save(trace_ad, trace_)
        np.save(pred_EQG_ad, pred_EQG_)
        np.save(pred_Mag_ad, pred_Mag_)
        np.save(pred_COI_ad, pred_COI_)
        np.save(pred_CRE_ad, pred_CRE_)

    return pos_, true_, trace_, pred_EQG_, pred_Mag_, pred_COI_, pred_CRE_


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
