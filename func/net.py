"""
Functions for modeling and graph construction
"""
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class MagNet(nn.Module):
    def __init__(self):
        super(MagNet, self).__init__()
        self.cnn1 = nn.Conv2d(1, 64, kernel_size=(1, 3), padding=(0, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4))
        self.drop = nn.Dropout(p=0.2)
        self.cnn2 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(0, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4))
        self.lstm = nn.LSTM(32, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(750, 1)

    def forward(self, x):
        h = self.cnn1(x.unsqueeze(1))
        h = self.pool1(self.drop(h))
        h = self.cnn2(h)
        h = self.pool2(self.drop(h))
        h = h.squeeze(2)
        h = h.permute(0, 2, 1)
        h, (_, _) = self.lstm(h)
        h = h.reshape(h.shape[0], -1)
        h = self.linear(h)
        return h.view(-1)


def cal_rmse_one_arr(true, pred):
    return np.sqrt(np.mean(np.square(true - pred)))


def cal_r2_one_arr(true, pred):
    corr_matrix = np.corrcoef(true, pred)
    corr = corr_matrix[0, 1]
    r2 = corr ** 2
    return r2


# generate graph neural network layers under different style
def gnn_layers(in_dim, hid_dim, out_dim, gnn_style):
    if gnn_style == "gcn":
        gnn_1 = gnn.GCNConv(in_dim, hid_dim)
        gnn_2 = gnn.GCNConv(hid_dim, out_dim)
    elif gnn_style == "cheb":
        gnn_1 = gnn.ChebConv(in_dim, hid_dim, K=2)
        gnn_2 = gnn.ChebConv(hid_dim, out_dim, K=2)
    elif gnn_style == "gin":
        gnn_1 = gnn.GraphConv(in_dim, hid_dim)
        gnn_2 = gnn.GraphConv(hid_dim, out_dim)
    elif gnn_style == "graphsage":
        gnn_1 = gnn.SAGEConv(in_dim, hid_dim)
        gnn_2 = gnn.SAGEConv(hid_dim, out_dim)
    elif gnn_style == "tag":
        gnn_1 = gnn.TAGConv(in_dim, hid_dim)
        gnn_2 = gnn.TAGConv(hid_dim, out_dim)
    elif gnn_style == "unimp":
        gnn_1 = gnn.TransformerConv(in_dim, hid_dim)
        gnn_2 = gnn.TransformerConv(hid_dim, out_dim)
    elif gnn_style == "gat":
        gnn_1 = gnn.GATConv(in_dim, hid_dim)
        gnn_2 = gnn.GATConv(hid_dim, out_dim)
    elif gnn_style == "arma":
        gnn_1 = gnn.ARMAConv(in_dim, hid_dim)
        gnn_2 = gnn.ARMAConv(hid_dim, out_dim)
    else:
        raise TypeError("Unknown Type of gnn_style!")
    return gnn_1, gnn_2


class CREIME(nn.Module):
    def __init__(self):
        super(CREIME, self).__init__()
        self.cnn1 = nn.Conv1d(3, 32, kernel_size=15, stride=1, padding=7)
        self.cnn2 = nn.Conv1d(32, 16, kernel_size=15, stride=1, padding=7)
        self.cnn3 = nn.Conv1d(16, 8, kernel_size=15, stride=1, padding=7)
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.lstm1 = nn.LSTM(8, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 256, batch_first=True)
        self.linear = nn.Linear(2048, 512)

    def forward(self, x):
        h = self.cnn1(x)
        h = self.pool(h)
        h = self.cnn2(h)
        h = self.pool(h)
        h = self.cnn3(h)
        h = self.pool(h)

        out, (_, _) = self.lstm1(h)
        out, (_, _) = self.lstm2(out)

        put = out.reshape(out.shape[0], -1)
        put = self.linear(put)
        return put


class MagPre(nn.Module):
    def __init__(self, gnn_style, adm_style, k, device):
        super(MagPre, self).__init__()
        self.relu = nn.ReLU()
        self.gnn_style = gnn_style
        self.adm_style = adm_style
        self.k, self.device = k, device
        self.pre = nn.Sequential(nn.ReLU())
        self.cnn1 = nn.Conv1d(3, 16, kernel_size=2, stride=2)
        self.cnn2 = nn.Conv1d(16, 16, kernel_size=2, stride=2)
        self.cnn3 = nn.Conv1d(16, 16, kernel_size=2, stride=2)
        self.cnn4 = nn.Conv1d(16, 32, kernel_size=2, stride=2)
        self.cnn5 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn6 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn7 = nn.Conv1d(32, 64, kernel_size=2, stride=2)
        self.cnn8 = nn.Conv1d(64, 64, kernel_size=2, stride=2)
        self.cnn9 = nn.Conv1d(64, 64, kernel_size=2, stride=2)
        self.cnn10 = nn.Conv1d(64, 128, kernel_size=2, stride=2)
        self.cnn11 = nn.Conv1d(128, 128, kernel_size=2, stride=2)
        self.linear = nn.Linear(256, 1)
        self.ei1, self.ew1 = get_edge_info(k, 3000, adm_style, device)
        self.ei2, self.ew2 = get_edge_info(k, 1500, adm_style, device)
        self.ei3, self.ew3 = get_edge_info(k, 750, adm_style, device)
        self.ei4, self.ew4 = get_edge_info(k, 375, adm_style, device)
        self.ei5, self.ew5 = get_edge_info(k, 187, adm_style, device)
        self.ei6, self.ew6 = get_edge_info(k, 93, adm_style, device)
        self.ei7, self.ew7 = get_edge_info(k, 46, adm_style, device)
        self.ei8, self.ew8 = get_edge_info(k, 23, adm_style, device)
        self.ei9, self.ew9 = get_edge_info(k, 11, adm_style, device)
        self.ei10, self.ew10 = get_edge_info(k, 5, adm_style, device)
        self.gnn1 = get_gnn(gnn_style, 16, 16)
        self.gnn2 = get_gnn(gnn_style, 16, 16)
        self.gnn3 = get_gnn(gnn_style, 16, 16)
        self.gnn4 = get_gnn(gnn_style, 32, 32)
        self.gnn5 = get_gnn(gnn_style, 32, 32)
        self.gnn6 = get_gnn(gnn_style, 32, 32)
        self.gnn7 = get_gnn(gnn_style, 64, 64)
        self.gnn8 = get_gnn(gnn_style, 64, 64)
        self.gnn9 = get_gnn(gnn_style, 64, 64)
        self.gnn10 = get_gnn(gnn_style, 128, 128)

    def forward(self, x):
        h_0 = h = self.cnn1(x)
        h = run_gnn(self.gnn_style, self.gnn1, h, self.ei1, self.ew1)
        h = h + h_0
        h_1 = h = self.cnn2(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn2, h, self.ei2, self.ew2)
        h = h + h_1
        h_2 = h = self.cnn3(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn3, h, self.ei3, self.ew3)
        h = h + h_2
        h_3 = h = self.cnn4(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn4, h, self.ei4, self.ew4)
        h = h + h_3
        h_4 = h = self.cnn5(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn5, h, self.ei5, self.ew5)
        h = h + h_4
        h_5 = h = self.cnn6(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn6, h, self.ei6, self.ew6)
        h = h + h_5
        h_6 = h = self.cnn7(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn7, h, self.ei7, self.ew7)
        h = h + h_6
        h_7 = h = self.cnn8(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn8, h, self.ei8, self.ew8)
        h = h + h_7
        h_8 = h = self.cnn9(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn9, h, self.ei9, self.ew9)
        h = h + h_8
        h_9 = h = self.cnn10(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn10, h, self.ei10, self.ew10)
        h = h + h_9
        h = self.cnn11(self.pre(h))

        out = h.view(h.shape[0], -1)
        out = self.linear(out)
        return out.view(-1)


class PSPick(nn.Module):
    def __init__(self, gnn_style, adm_style, k, device):
        super(PSPick, self).__init__()
        self.relu = nn.ReLU()
        self.gnn_style = gnn_style
        self.adm_style = adm_style
        self.k, self.device = k, device
        self.pre = nn.Sequential(nn.ReLU())
        self.cnn1 = nn.Conv1d(3, 16, kernel_size=2, stride=2)
        self.cnn2 = nn.Conv1d(16, 16, kernel_size=2, stride=2)
        self.cnn3 = nn.Conv1d(16, 16, kernel_size=2, stride=2)
        self.cnn4 = nn.Conv1d(16, 32, kernel_size=2, stride=2)
        self.cnn5 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn6 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn7 = nn.Conv1d(32, 64, kernel_size=2, stride=2)
        self.cnn8 = nn.Conv1d(64, 64, kernel_size=2, stride=2)
        self.cnn9 = nn.Conv1d(64, 64, kernel_size=2, stride=2)
        self.cnn10 = nn.Conv1d(64, 128, kernel_size=2, stride=2)
        self.cnn11 = nn.Conv1d(128, 128, kernel_size=2, stride=2)
        self.linear = nn.Linear(256, 2)
        self.ei1, self.ew1 = get_edge_info(k, 3000, adm_style, device)
        self.ei2, self.ew2 = get_edge_info(k, 1500, adm_style, device)
        self.ei3, self.ew3 = get_edge_info(k, 750, adm_style, device)
        self.ei4, self.ew4 = get_edge_info(k, 375, adm_style, device)
        self.ei5, self.ew5 = get_edge_info(k, 187, adm_style, device)
        self.ei6, self.ew6 = get_edge_info(k, 93, adm_style, device)
        self.ei7, self.ew7 = get_edge_info(k, 46, adm_style, device)
        self.ei8, self.ew8 = get_edge_info(k, 23, adm_style, device)
        self.ei9, self.ew9 = get_edge_info(k, 11, adm_style, device)
        self.ei10, self.ew10 = get_edge_info(k, 5, adm_style, device)
        self.gnn1 = get_gnn(gnn_style, 16, 16)
        self.gnn2 = get_gnn(gnn_style, 16, 16)
        self.gnn3 = get_gnn(gnn_style, 16, 16)
        self.gnn4 = get_gnn(gnn_style, 32, 32)
        self.gnn5 = get_gnn(gnn_style, 32, 32)
        self.gnn6 = get_gnn(gnn_style, 32, 32)
        self.gnn7 = get_gnn(gnn_style, 64, 64)
        self.gnn8 = get_gnn(gnn_style, 64, 64)
        self.gnn9 = get_gnn(gnn_style, 64, 64)
        self.gnn10 = get_gnn(gnn_style, 128, 128)

    def forward(self, x):
        h_0 = h = self.cnn1(x)
        h = run_gnn(self.gnn_style, self.gnn1, h, self.ei1, self.ew1)
        h = h + h_0
        h_1 = h = self.cnn2(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn2, h, self.ei2, self.ew2)
        h = h + h_1
        h_2 = h = self.cnn3(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn3, h, self.ei3, self.ew3)
        h = h + h_2
        h_3 = h = self.cnn4(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn4, h, self.ei4, self.ew4)
        h = h + h_3
        h_4 = h = self.cnn5(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn5, h, self.ei5, self.ew5)
        h = h + h_4
        h_5 = h = self.cnn6(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn6, h, self.ei6, self.ew6)
        h = h + h_5
        h_6 = h = self.cnn7(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn7, h, self.ei7, self.ew7)
        h = h + h_6
        h_7 = h = self.cnn8(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn8, h, self.ei8, self.ew8)
        h = h + h_7
        h_8 = h = self.cnn9(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn9, h, self.ei9, self.ew9)
        h = h + h_8
        h_9 = h = self.cnn10(self.pre(h))
        h = run_gnn(self.gnn_style, self.gnn10, h, self.ei10, self.ew10)
        h = h + h_9
        h = self.cnn11(self.pre(h))

        out = h.view(h.shape[0], -1)
        out = self.linear(out)
        return out


def get_edge_info(k, num_nodes, adm_style, device):
    if adm_style == "ts_un":
        adm = ts_un(num_nodes, k)
    elif adm_style == "tg":
        adm = tg(num_nodes)
    else:
        raise TypeError("Unknown type of adm_style!")
    edge_index, edge_weight = tran_adm_to_edge_index(adm)
    edge_index = edge_index.to(device)
    return edge_index, nn.Parameter(edge_weight)


def get_gnn(gnn_style, in_dim, out_dim):
    if gnn_style == "gcn":
        return gnn.GCNConv(in_dim, out_dim)
    elif gnn_style == "cheb":
        return gnn.ChebConv(in_dim, out_dim, K=1)
    elif gnn_style == "gin":
        return gnn.GraphConv(in_dim, out_dim)
    elif gnn_style == "graphsage":
        return gnn.SAGEConv(in_dim, out_dim)
    elif gnn_style == "tag":
        return gnn.TAGConv(in_dim, out_dim)
    elif gnn_style == "sg":
        return gnn.SGConv(in_dim, out_dim)
    elif gnn_style == "appnp":
        return gnn.APPNP(K=2, alpha=0.5)
    elif gnn_style == "arma":
        return gnn.ARMAConv(in_dim, out_dim)
    elif gnn_style == "cg":
        return gnn.CGConv(in_dim)
    else:
        raise TypeError("Unknown type of gnn_style!")


def run_gnn(gnn_style, gnn, x, ei, ew):
    if gnn_style in ["gcn", "cheb", "sg", "appnp"]:
        return gnn(x.permute(0, 2, 1), ei, ew).permute(0, 2, 1)
    else:
        return gnn(x.permute(0, 2, 1), ei).permute(0, 2, 1)


class ConvNetQuakeINGV(nn.Module):
    def __init__(self):
        super(ConvNetQuakeINGV, self).__init__()
        self.relu = nn.ReLU()
        self.cnn1 = nn.Conv1d(3, 32, kernel_size=2, stride=2)
        self.cnn2 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn3 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn4 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn5 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn6 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn7 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn8 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn9 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.linear = nn.Linear(352, 127)

    def forward(self, x):
        h = self.cnn1(x)
        h = self.cnn2(self.relu(h))
        h = self.cnn3(self.relu(h))
        h = self.cnn4(self.relu(h))
        h = self.cnn5(self.relu(h))
        h = self.cnn6(self.relu(h))
        h = self.cnn7(self.relu(h))
        h = self.cnn8(self.relu(h))
        h = self.cnn9(self.relu(h))

        out = h.view(h.shape[0], -1)
        out = self.linear(out)
        out = torch.mean(out[:, 52:72], dim=1)
        return out


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


def tg(m):
    adm = np.zeros(shape=(m, m))
    for i in range(m - 1):
        adm[i + 1, i] = 1
    adm[0, m - 1] = 1
    adm = adm * 0.5
    return adm


def hvg(m, x):
    u_max = 0
    adm = np.zeros(shape=(m, m))
    for i in range(m):
        if i == (m - 1):                    # 最后一个节点
            continue
        for j in range(i + 1, m):
            if j == (i + 1):                # 相邻点必相连
                adm[i, j], adm[j, i] = 1., 1.
                u_max = x[j]
                if x[j] > x[i]:
                    break
            else:
                if x[j] < u_max:
                    continue
                elif x[j] > u_max:
                    adm[i, j], adm[j, i] = 1., 1.
                    if x[j] < x[i]:
                        u_max = x[j]
                    else:
                        break
    adm = adm * 0.5
    return adm


def nvg(m, x):
    u_max = 0
    adm = np.zeros(shape=(m, m))
    for i in range(m):
        if i == (m - 1):
            break
        for j in range(i + 1, m):
            if j == (i + 1):
                u_max = x[j]
                adm[i, j], adm[j, i] = 1., 1.
            else:
                if (x[j] - u_max) > (u_max - x[i]):
                    u_max = x[j]
                    adm[i, j], adm[j, i] = 1., 1.
                else:
                    pass
    adm = adm * 0.5
    return adm


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
