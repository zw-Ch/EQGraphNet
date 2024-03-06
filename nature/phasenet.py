import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

a = sbg.WindowAroundSample(["trace_name"], samples_before=3000, windowlen=6000,
                           selection="random", strategy="variable")

model = sbm.PhaseNet(phases="PSN")
phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_Pl_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_Sl_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}

# Get dataset and split
data = sbd.InstanceCountsCombined(sampling_rate=100)
train, dev, test = data.train_dev_test()

# add preprocessing and data augmentation steps
train_generator = sbg.GenericGenerator(train)
dev_generator = sbg.GenericGenerator(dev)

augmentations = [
    sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000,
                           selection="random", strategy="variable"),
    sbg.RandomWindow(windowlen=3001, strategy="pad"),
    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    sbg.ChangeDtype(np.float32),
    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0)
]

train_generator.add_augmentations(augmentations)
dev_generator.add_augmentations(augmentations)

batch_size = 256
num_workers = 4

train_loader = DataLoader(train_generator, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, worker_init_fn=worker_seeding)
dev_loader = DataLoader(dev_generator, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, worker_init_fn=worker_seeding)

learning_rate = 1e-2
epochs = 5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def loss_fn(y_pred, y_true, eps=1e-5):
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)
    h = h.mean()
    return -h


def train_loop(dataloader):
    size = len(dataloader.dataset)

    for batch_id, batch in enumerate(dataloader):
        pred = model(batch["X"].to(model.device))
        loss = loss_fn(pred, batch["y"].to(model.device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 5 == 0:
            loss, current = loss.item(), batch_id * batch["X"].shape[0]
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}")


def dev_loop(dataloader):
    num_batches = len(dataloader)
    dev_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch["X"].to(model.device))
            dev_loss += loss_fn(pred, batch["y"].to(model.device)).item()

    dev_loss /= num_batches
    print(f"Test avg loss: {dev_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t + 1}\n")
    train_loop(train_loader)
    dev_loop(dev_loader)

print()
