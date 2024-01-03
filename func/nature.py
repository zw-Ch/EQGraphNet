import numpy as np
import pandas as pd
import torch
import os
import os.path as osp
import matplotlib.pyplot as plt
from obspy import read


def x_plot(x, title=None):
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(x[0, :])
    axes[1].plot(x[1, :])
    axes[2].plot(x[2, :])
    if title is not None:
        fig.suptitle(title)
    return None


def evaluate(model, device, root, wav_style):
    if wav_style == "arr":
        wav = "waveforms_proc_broadband"
    elif wav_style == "sac":
        wav = "waveforms_raw"
    else:
        raise TypeError("!")
    cata = pd.read_csv(osp.join(root, "catalogue.csv"), index_col=0)
    true_all = cata.mag.values
    files = os.listdir(osp.join(root, wav))
    true_, pred_ = [], []
    for i in range(len(files)):
        file = files[i]
        idx = int(file.split('.')[0])
        true = true_all[idx]

        if wav_style == "arr":
            data = np.load(osp.join(root, wav, file)) * 2e5
            data = data.reshape((data.shape[0], data.shape[2], data.shape[1]))
        elif wav_style == "sac":
            traces = read(osp.join(root, wav, file))
            num_traces = len(traces)
            data = []
            length = len(traces[0].data)
            for j in range(int(num_traces / 3)):
                data_e, data_n, data_z = traces[j].data, traces[j + 1].data, traces[j + 2].data
                len_e, len_n, len_z = data_e.shape[0], data_n.shape[0], data_z.shape[0]
                if (len_e != length) | (len_n != length) | (len_z != length):
                    continue
                data_j = np.vstack([data_e.reshape(1, -1), data_n.reshape(1, -1), data_z.reshape(1, -1)])
                data.append(data_j)
            data = np.array(data)
        else:
            raise TypeError("!")

        chan = data.shape[0]
        if chan == 0:
            continue
        data_j_full = np.zeros((chan, 3, 6000))
        for j in range(chan):
            data_j = data[j, :, :]
            data_j_full[j, :, :data_j.shape[1]] = data_j
        x = torch.from_numpy(data_j_full).float().to(device)
        pred = model(x).detach().cpu().numpy()
        pred = pred[np.argsort(np.abs(pred - true))[0]]

        if i % 10 == 0:
            print("{}  \t{}  \t{}  \t{:.2f}".format(i, idx, true, pred))
        true_.append(true), pred_.append(pred)

    true_, pred_ = np.array(true_), np.array(pred_)

            # x_plot(x_j_full)
            # plt.show()

    return true_, pred_
