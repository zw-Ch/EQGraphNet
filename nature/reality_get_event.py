"""
获取震源信息（真实）
"""
import pandas as pd
from obspy.core.event import read_events
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os.path as osp


client_list = ["INGV", "IRIS"]
num_client = len(client_list)
start_time = UTCDateTime("2009-01-01T00:00:00")
end_time = UTCDateTime("2020-01-01T00:00:00")

min_lat, max_lat, min_lon, max_lon = 35, 45, 10, 20
min_mag = 3
min_depth = 1

event_all = []
for i in range(num_client):
    client = Client(client_list[i])
    try:
        event = client.get_events(
            minlatitude=min_lat, maxlatitude=max_lat, minlongitude=min_lon, maxlongitude=max_lon,
            starttime=start_time, endtime=end_time, mindepth=min_depth, minmagnitude=min_mag)
    except:
        continue
    # event.plot(projection="local")
    if len(event_all) == 0:
        event_all = event
    else:
        event_all = event_all + event

events = event_all.events
num_event = len(events)
mag, scale, lat, lon, time = [], [], [], [], []
for i in range(num_event):
    event_one = events[i]                   # 单独一个地震事件event
    info = event_one.short_str()            # 提取出event的描述信息

    info_ = info.split(',')
    info_ = info_[0] + info_[1]

    info_ = info_.split(' ')

    num_remove, idx = info_.count('|'), 0           # 除去'|'
    while idx < num_remove:
        info_.remove('|')
        idx = idx + 1

    num_remove, idx = info_.count(''), 0           # 除去''
    while idx < num_remove:
        info_.remove('')
        idx = idx + 1

    if len(info_) != 6:
        continue

    mag_one, scale_one = float(info_[3]), info_[4]  # 震级、地震种类
    time_one, lat_one, lon_one = info_[0], float(info_[1]), float(info_[2])      # 纬度、经度

    if scale_one not in ["ML", "ml", "MD", "md"]:
        continue

    mag.append(mag_one), scale.append(scale_one)
    time.append(time_one), lat.append(lat_one), lon.append(lon_one)

mag, scale = np.array(mag).reshape(-1, 1), np.array(scale).reshape(-1, 1)
time, lat, lon = np.array(time).reshape(-1, 1), np.array(lat).reshape(-1, 1), np.array(lon).reshape(-1, 1)

event_info = np.hstack([mag, scale, time, lat, lon])
event_info = pd.DataFrame(event_info, columns=["Magnitude", "Scale", "Time", "Latitude", "Longitude"])
event_info.to_csv(osp.join("../factor/reality_data/event_info.csv"))

# # 记录震源的相关信息，并保存
# event_info = {'loc': loc, 'mag': mag, 'scale': scale}
# event_info_path = "reality_data/event_info.pkl"
# with open(event_info_path, 'wb') as fp:
#     pickle.dump(event_info, fp)

print()
