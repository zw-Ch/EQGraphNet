"""
获取地震信号（真实）
"""
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import os.path as osp


def x_plot(st_e, st_n, st_z, title):
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(st_e.traces[0].data)
    axes[1].plot(st_n.traces[0].data)
    axes[2].plot(st_z.traces[0].data)
    fig.suptitle(title)
    return None


def get_time_start(time):
    time_ = time.split(":")[0]
    time_ = time_.split("T")
    hour = int(time_[-1])
    time_ = time_[0].split("-")
    year, mon, day = int(time_[0]), int(time_[1]), int(time_[2])
    return year, mon, day, hour


def judge(st):
    if st is None:
        return False
    else:
        x = st.traces[0].data[:6000]
        length = np.argwhere(x < 1).reshape(-1).shape[0]
        if length > 5000:
            return False
        else:
            return True


def get_signal(net, sta, chan, time_begin, time_end):
    # print(net, sta, chan)
    trace = osp.join(dir_data, net + "." + sta + "." + chan)
    if os.path.exists(trace):
        print(net, sta, chan, " downloaded already")
        return None, trace
    try:
        st = client1.get_waveforms(
            network=net, station=sta, location=False, channel=chan, starttime=time_begin,
            endtime=time_end, attach_response=True)
        # st.merge(method=1, fill_value="interpolate")
        # st_length = st.traces[0].data.shape[0]
        st.resample(5 / 6)
        st = st.trim(time_begin, time_end, pad=True, fill_value=0)
        st.detrend("demean")
        st.detrend("linear")
        print(net, sta, chan, " have been downloaded")
        return st, trace
    except:
        try:
            st = client2.get_waveforms(
                network=net, station=sta, location=False, channel=chan, starttime=time_begin,
                endtime=time_end, attach_response=True)
            # st.merge(method=1, fill_value="interpolate")
            # st_length = st.traces[0].data.shape[0]
            st.resample(5 / 6)
            st = st.trim(time_begin, time_end, pad=True, fill_value=0)
            st.detrend("demean")
            st.detrend("linear")
            print(net, sta, chan, " have been downloaded")
            return st, trace
        except:
            print(net, sta, chan, " doesn't exist")
            return None, trace


"""
downloading earthquake signals
"""
event_info = pd.read_csv("reality_data/event_info.csv", index_col=0)

time_start = "2019-12-09T05:19:33.940000Z"          # 地震发生时刻（从event_info中手动选择）
year, mon, day, hour = get_time_start(time_start)

station_file = "reality_data/station.dat"
source1 = "INGV"
client1 = Client(source1)
source2 = "IRIS"
client2 = Client(source2)
dir_data = "reality_data/{}_{}_{}".format(year, mon, day)

if not os.path.exists(dir_data):
    os.mkdir(dir_data)

time_origin = UTCDateTime(year, mon, day) + timedelta(seconds=(hour * 3600))
time_begin = time_origin - timedelta(seconds=0)          # 地震信号，开始采集时刻（地震开始时）
time_end = time_origin + timedelta(seconds=7200)         # 地震信号，终止采集时刻（地震开始后，2个小时）

f = open(station_file, "r")
for station in f:
    sta_lo, sta_la, net, sta, chan, elev = station.split()
    chan_e, chan_n, chan_z = chan[:2] + "E", chan[:2] + "N", chan[:2] + "Z"
    st_e, trace_e = get_signal(net, sta, chan_e, time_begin, time_end)
    st_n, trace_n = get_signal(net, sta, chan_n, time_begin, time_end)
    st_z, trace_z = get_signal(net, sta, chan_z, time_begin, time_end)
    if judge(st_e) & judge(st_n) & (judge(st_z)):
        st_e.write(filename=trace_e, format="SAC")
        st_n.write(filename=trace_n, format="SAC")
        st_z.write(filename=trace_z, format="SAC")
        x_plot(st_e, st_n, st_z, "{}_{}_{}".format(net, sta, chan))
    else:
        print(net, sta, chan, " 不满足要求")

f.close()


print()
plt.show()
print()
