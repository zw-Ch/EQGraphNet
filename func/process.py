"""
Functions for processing and calculating data
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
import os
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Chunk(Dataset):
    def __init__(self, num, train, num_train, idx, root, name):
        super(Chunk, self).__init__()
        self.num, self.root, self.name = num, root, name
        self.save_ad = osp.join(root, str(num))
        self.df = pd.read_csv(osp.join(self.root, self.name + ".csv"))
        self.data, self.index = self.get_sample()
        self.df = self.df.iloc[self.index, :]
        self.num_train = num_train
        self.length = self.data.shape[2]
        self.train = train
        self.idx = idx
        self.get_train_or_test()

    def get_train_or_test(self):
        self.data = self.data[self.idx, :, :]
        self.index = self.index[self.idx]
        self.df = self.df.iloc[self.idx, :]
        return None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data, index = self.data[idx, :, :], self.index[idx]
        return data, index

    def get_sample(self):
        if not osp.exists(self.save_ad):
            os.makedirs(self.save_ad)
        data_ad = osp.join(self.save_ad, "data.pt")
        index_ad = osp.join(self.save_ad, "index.pt")
        if osp.exists(data_ad) & osp.exists(index_ad):
            data = torch.load(data_ad)
            index = torch.load(index_ad)
        else:
            metadata = h5py.File(osp.join(self.root, self.name + ".hdf5"), 'r')

            trace_name = self.df.loc[:, "trace_name"].values.reshape(-1)
            index = np.random.choice(trace_name.shape[0], self.num, replace=False).tolist()

            ev_list = self.df['trace_name'].to_list()
            data = np.zeros(shape=(self.num, 3, 6000))
            for c, i in enumerate(index):
                ev_one = ev_list[i]
                dataset_one = metadata.get('data/' + str(ev_one))
                data_one = np.array(dataset_one)
                data_one = np.expand_dims(data_one.T, axis=0)
                data[c, :, :] = data_one

            data = torch.from_numpy(data).float()
            index = torch.FloatTensor(index).int()

            torch.save(data, data_ad)
            torch.save(index, index_ad)
        return data, index


def get_train_or_test_idx(num, num_train):
    idx_all = np.arange(num)
    idx_train = np.random.choice(num, num_train, replace=False)
    idx_test = np.array(list(set(idx_all) - set(idx_train)))
    return idx_train, idx_test


def be_tensor(x):
    if type(x) == np.ndarray:
        return torch.from_numpy(x)
    elif torch.is_tensor(x):
        return x
    else:
        raise TypeError("x must be tensor or ndarray, but gut {}".format(type(x)))


def get_item_by_dim(data, item):
    if torch.is_tensor(data):
        n_dim = data.dim()
    elif type(data) == np.ndarray:
        n_dim = data.ndim
    else:
        raise TypeError("The input must be torch.tensor or numpy.ndarray!")
    if n_dim == 1:
        return data[item]
    elif n_dim == 2:
        return data[item, :]
    elif n_dim == 3:
        return data[item, :, :]
    elif n_dim == 4:
        return data[item, :, :, :]
    else:
        raise ValueError("Unknown dim() of input!")


class SelfData(Dataset):
    def __init__(self, data, label, *args):
        super(SelfData, self).__init__()
        self.data = be_tensor(data)
        self.label = be_tensor(label)
        self.args = args
        self.data_else = self.get_data_else()

    def get_data_else(self):
        num = len(self.args)
        data_else = [0] * num
        if num != 0:
            for i in range(num):
                data_else_one = self.args[i]
                data_else[i] = data_else_one
        return data_else

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        data_one = get_item_by_dim(self.data, item)
        label_one = get_item_by_dim(self.label, item)
        result = [data_one, label_one]
        if len(self.data_else) != 0:
            num = len(self.data_else)
            data_else_one = [0] * num
            for i in range(num):
                x = self.data_else[i]
                x_one = get_item_by_dim(x, item)
                data_else_one[i] = x_one
            result = result + data_else_one
        result.append(item)
        return tuple(result)


def ts_un(n, k):
    adm = np.zeros(shape=(n, n))
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")
    else:
        for i in range(n):
            if i < (n - k):
                for k_one in range(1, k + 1):
                    adm[i, i + k_one] = 1.
            else:
                for k_one in range(1, k + 1):
                    if (k_one + i) >= n:
                        mod = (k_one + i) % n
                        adm[i, mod] = 1.
                    else:
                        adm[i, i + k_one] = 1.
    adm = (adm.T + adm) / 2
    # adm = adm * 0.5
    return adm


# processing data and return preprocessor
def prep_tran(prep_style, data, *args):
    if prep_style == "sta":
        model = StandardScaler()
    elif prep_style == "min":
        model = MinMaxScaler()
    else:
        raise TypeError("Unknown Type of prep_style!")
    model.fit(data)
    data_prep = model.transform(data)
    data_prep = data_prep.reshape(data.shape)
    num, data_prep_other = len(args), [0] * len(args)
    if num == 0:
        return model, data_prep
    for i in range(num):
        one = args[i]
        one_tran = model.transform(one)
        one_tran = one_tran.reshape(one.shape)
        data_prep_other[i] = one_tran
    if num == 1:
        data_prep_other = data_prep_other[0]
    else:
        data_prep_other = tuple(data_prep_other)
    return model, data_prep, data_prep_other


# recover preprocessed data by given preprocessor
def prep_inv(model, *args):
    num = len(args)
    if num == 0:
        raise ValueError("Please input data for inverse-normalization!")
    inv = [0] * num
    for i in range(num):
        one = args[i]
        one_inv = model.inverse_transform(one)
        one_inv = one_inv.reshape(one.shape)
        inv[i] = one_inv
    if num == 1:
        inv = inv[0]
    else:
        inv = tuple(inv)
    return inv


# preprocessing P and S wave arrival time
def prep_pt(prep_style, train, test=None):
    if prep_style == "sta":
        model = StandardScaler()
    else:
        raise TypeError("Unknown Type of prep_style!")
    if train.ndim == 1:
        train = train.reshape(-1, 1)
    model.fit(train)
    train_prep = model.transform(train)
    if test is None:
        return model, train_prep
    if test.ndim == 1:
        test = test.reshape(-1, 1)
    test_prep = model.transform(test)
    return model, train_prep, test_prep


# for torch_geometric.nn
def tran_adm_to_edge_index(adm):
    u, v = np.nonzero(adm)
    num_edges = u.shape[0]
    edge_index = np.vstack([u.reshape(1, -1), v.reshape(1, -1)])
    edge_weight = np.zeros(shape=u.shape)
    for i in range(num_edges):
        edge_weight_one = adm[u[i], v[i]]
        edge_weight[i] = edge_weight_one
    edge_index = torch.from_numpy(edge_index).long()
    edge_weight = torch.from_numpy(edge_weight).float()
    return edge_index, edge_weight


def remain_sm_scale(data, df, label, scale):
    if isinstance(scale, list):
        smt = df['source_magnitude_type'].isin(scale).values
        idx = np.argwhere(smt).reshape(-1)
        num = len(scale)
        scale_name = ""
        for i in range(num):
            if i == 0:
                scale_name = scale[i]
            else:
                scale_name = scale_name + "_" + scale[i]
    else:
        smt = df.source_magnitude_type.values.reshape(-1)
        idx = np.argwhere(smt == scale).reshape(-1)
        scale_name = scale
    data = data[idx, :, :]
    label = label[idx]
    df = df.iloc[idx, :]
    return data, label, df, scale_name


def add_noise(data_train, data_test, sm_train, sm_test, root_no, name_no, m, thre):
    m_train, m_test = sm_train.shape[0], sm_test.shape[0]
    idx_train, idx_test = get_train_or_test_idx(m_train + m_test, m_train)
    no_train = Chunk(m, True, m_train, idx_train, root_no, name_no)
    no_test = Chunk(m, False, m_train, idx_test, root_no, name_no)
    no_train_data, no_test_data = no_train.data, no_test.data
    no_train_sm, no_test_sm = torch.ones(m_train).float() * thre, torch.ones(m_test).float() * thre

    train_data = torch.cat((data_train, no_train_data), dim=0)
    test_data = torch.cat((data_test, no_test_data), dim=0)
    train_sm = torch.cat((sm_train, no_train_sm), dim=0)
    test_sm = torch.cat((sm_test, no_test_sm), dim=0)
    return train_data, test_data, train_sm, test_sm


def save_result(re_ad, model, save_np, save_model, save_loss, sm_scale, name, m_train, m_test,
                train_true, train_pred, train_trace, train_pos, train_loss,
                test_true, test_pred, test_trace, test_pos, test_loss):
    if save_np:
        np.save(osp.join(re_ad, "train_true_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), train_true)
        np.save(osp.join(re_ad, "train_pred_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), train_pred)
        np.save(osp.join(re_ad, "train_trace_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), train_trace)
        np.save(osp.join(re_ad, "train_pos_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), train_pos)
        np.save(osp.join(re_ad, "test_true_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), test_true)
        np.save(osp.join(re_ad, "test_pred_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), test_pred)
        np.save(osp.join(re_ad, "test_trace_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), test_trace)
        np.save(osp.join(re_ad, "test_pos_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), test_pos)
    if save_model:
        torch.save(model.state_dict(),
                   osp.join(re_ad, "model_{}_{}_{}_{}.pkl".format(sm_scale, name, m_train, m_test)))
    if save_loss:
        np.save(osp.join(re_ad, "train_loss_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), train_loss)
        np.save(osp.join(re_ad, "test_loss_{}_{}_{}_{}.npy".format(sm_scale, name, m_train, m_test)), test_loss)
    return True


def read_coda(df):
    coda = df['coda_end_sample'].values.reshape(-1)

    def tran_one(element):
        return int(element.split('[')[-1].split('.')[0])

    coda = list(tran_one(item) for item in coda)
    return coda


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


def get_mai_data(df_train, df_test):
    ps_at_name = ["p_arrival_sample", "s_arrival_sample"]
    ps_at_train, ps_at_test = df_train.loc[:, ps_at_name].values, df_test.loc[:, ps_at_name].values
    prep_ps_at, ps_at_train, ps_at_test = prep_pt("sta", ps_at_train, ps_at_test)
    ps_at_train, ps_at_test = torch.from_numpy(ps_at_train).float(), torch.from_numpy(ps_at_test).float()

    t_name = ["p_travel_sec"]
    p_t_train, p_t_test = df_train.loc[:, t_name].values, df_test.loc[:, t_name].values
    prep_p_t, p_t_train, p_t_test = prep_pt("sta", p_t_train, p_t_test)
    p_t_train, p_t_test = torch.from_numpy(p_t_train).float(), torch.from_numpy(p_t_test).float()
    return ps_at_train, ps_at_test, p_t_train, p_t_test


def get_pos_trace(df_train, df_test):
    pos_train = df_train.loc[:, ["source_longitude", "source_latitude"]].values
    pos_test = df_test.loc[:, ["source_longitude", "source_latitude"]].values
    trace_train = df_train['trace_name'].values.reshape(-1)
    trace_test = df_test['trace_name'].values.reshape(-1)
    return pos_train, pos_test, trace_train, trace_test


def get_loader(bz, name, root, m, sm_scale, train_ratio, random, style):
    m_train = int(m * train_ratio)              # number of training samples

    if not random:
        np.random.seed(100)
    idx_train, idx_test = get_train_or_test_idx(m, m_train)
    eq_train = Chunk(m, True, m_train, idx_train, root, name)
    eq_test = Chunk(m, False, m_train, idx_test, root, name)
    df_train, df_test = eq_train.df, eq_test.df

    data_train, data_test = eq_train.data.float(), eq_test.data.float()
    sm_train = torch.from_numpy(df_train["source_magnitude"].values.reshape(-1)).float()
    sm_test = torch.from_numpy(df_test["source_magnitude"].values.reshape(-1)).float()

    # Select samples according to Magnitude Type
    data_train, sm_train, df_train, sm_scale_name = remain_sm_scale(data_train, df_train, sm_train, sm_scale)
    data_test, sm_test, df_test, _ = remain_sm_scale(data_test, df_test, sm_test, sm_scale)

    if style == "mai_po_tr":
        pos_train, pos_test, trace_train, trace_test = get_pos_trace(df_train, df_test)
        ps_at_train, ps_at_test, p_t_train, p_t_test = get_mai_data(df_train, df_test)
        train_dataset = SelfData(data_train, sm_train, ps_at_train, p_t_train, pos_train, trace_train)
        test_dataset = SelfData(data_test, sm_test, ps_at_test, p_t_test, pos_test, trace_test)
    elif style == "mai":
        ps_at_train, ps_at_test, p_t_train, p_t_test = get_mai_data(df_train, df_test)
        train_dataset = SelfData(data_train, sm_train, ps_at_train, p_t_train)
        test_dataset = SelfData(data_test, sm_test, ps_at_test, p_t_test)
    elif style == "cre_po_tr":
        pos_train, pos_test, trace_train, trace_test = get_pos_trace(df_train, df_test)
        x_train, y_train = get_xy(data_train, df_train, sm_train, 125)
        x_test, y_test = get_xy(data_test, df_test, sm_test, 125)
        train_dataset = SelfData(x_train, y_train, sm_train, pos_train, trace_train)
        test_dataset = SelfData(x_test, y_test, sm_test, pos_test, trace_test)
    elif style == "cre":
        x_train, y_train = get_xy(data_train, df_train, sm_train, 125)
        x_test, y_test = get_xy(data_test, df_test, sm_test, 125)
        train_dataset = SelfData(x_train, y_train, sm_train)
        test_dataset = SelfData(x_test, y_test, sm_test)
    elif style == "tr_po":
        pos_train, pos_test, trace_train, trace_test = get_pos_trace(df_train, df_test)
        train_dataset = SelfData(data_train, sm_train, pos_train, trace_train)
        test_dataset = SelfData(data_test, sm_test, pos_test, trace_test)
    elif style == "":
        train_dataset = SelfData(data_train, sm_train)
        test_dataset = SelfData(data_test, sm_test)
    else:
        raise TypeError("Unknown type of 'style'!")
    train_loader = DataLoader(train_dataset, batch_size=bz, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bz, shuffle=True)
    return train_loader, test_loader, sm_scale_name
