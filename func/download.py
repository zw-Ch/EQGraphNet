import os
import os.path as osp
import json
import platform
from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client


def makeStationList(json_path, client_list, min_lat, max_lat, min_lon, max_lon, start_time, end_time, channel_list=[],
                    filter_network=[], filter_station=[],  **kwargs):
    """

    Uses fdsn to find available stations in a specific geographical location and time period.

    Parameters
    ----------
    json_path: str
        Path of the json file that will be returned

    client_list: list
        List of client names e.g. ["IRIS", "SCEDC", "USGGS"].

    min_lat: float
        Min latitude of the region.

    max_lat: float
        Max latitude of the region.

    min_lon: float
        Min longitude of the region.

    max_lon: float
        Max longitude of the region.

    start_time: str
        Start DateTime for the beginning of the period in "YYYY-MM-DDThh:mm:ss.f" format.

    end_time: str
        End DateTime for the beginning of the period in "YYYY-MM-DDThh:mm:ss.f" format.

    channel_list: str, default=[]
        A list containing the desired channel codes. Downloads will be limited to these channels based on priority. Defaults to [] --> all channels

    filter_network: str, default=[]
        A list containing the network codes that need to be avoided.

    filter_station: str, default=[]
        A list containing the station names that need to be avoided.

    kwargs:
        special symbol for passing Client.get_stations arguments

    Returns
    ----------
    stations_list.json: A dictionary containing information for the available stations.
    """
    station_list = {}
    for cl in client_list:
        inventory = Client(cl).get_stations(minlatitude=min_lat,
                                            maxlatitude=max_lat,
                                            minlongitude=min_lon,
                                            maxlongitude=max_lon,
                                            starttime=UTCDateTime(start_time),
                                            endtime=UTCDateTime(end_time),
                                            level='channel', **kwargs)
        for ev in inventory:
            net = ev.code
            if net not in filter_network:
                for st in ev:
                    station = st.code
                    print(str(net) + "--" + str(station))

                    if station not in filter_station:

                        elv = st.elevation
                        lat = st.latitude
                        lon = st.longitude
                        new_chan = [ch.code for ch in st.channels]
                        if len(channel_list) > 0:
                            chan_priority = [ch[:2] for ch in channel_list]

                            for chnn in chan_priority:
                                if chnn in [ch[:2] for ch in new_chan]:
                                    new_chan = [ch for ch in new_chan if ch[:2] == chnn]
# =============================================================================
#                      if ("BHZ" in new_chan) and ("HHZ" in new_chan):
#                          new_chan = [ch for ch in new_chan if ch[:2] != "BH"]
#                      if ("HHZ" in new_chan) and ("HNZ" in new_chan):
#                          new_chan = [ch for ch in new_chan if ch[:2] != "HH"]
#
#                           if len(new_chan)>3 and len(new_chan)%3 != 0:
#                               chan_type = [ch for ch in new_chan if ch[2] == 'Z']
#                               chan_groups = []
#                               for i, cht in enumerate(chan_type):
#                                   chan_groups.append([ch for ch in new_chan if ch[:2] == cht[:2]])
#                               new_chan2 = []
#                               for chg in chan_groups:
#                                   if len(chg) == 3:
#                                       new_chan2.append(chg)
#                               new_chan = new_chan2
# =============================================================================
                        new_chan_one = new_chan[0]
                        for i in range(len(st.channels)):
                            channel_one = st.channels[i].code
                            if channel_one == new_chan_one:
                                location = st.channels[i].location_code
                                break
                        if len(new_chan) > 0 and (station not in station_list):
                            station_list[str(station)] = {"network": net,
                                                          "channels": list(set(new_chan)),
                                                          "coords": [lat, lon, elv],
                                                          "location": location}
    json_dir = os.path.dirname(json_path)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    with open(json_path, 'w') as fp:
        json.dump(station_list, fp)
    return station_list


# 将station_list.json文件，生成用于LOC-FLOW的station.dat文件
def makeDatFile(json_path, file_name):
    json_file = open(json_path)
    station_dic = json.load(json_file)
    num_sta = len(station_dic)              # 台站的数量
    if platform.system() == "Windows":
        jf_ = json_path.split("\\")[1:-1]
    else:
        jf_ = json_path.split("/")[1:-1]
    folder_address = "/"
    for jf_one in jf_:                          # 保存station信息的文件夹
        folder_address = osp.join(folder_address, jf_one)
    dat_file = osp.join(folder_address, file_name)
    sta_idx = 0                     # 已统计的台站的数量
    with open(dat_file, 'w') as f:
        for i in station_dic.keys():
            station = station_dic[i]            # 某个台站的相关信息
            net = station['network']                # 网络
            channels = station['channels']          # 通道
            coords = station['coords']             # 位置
            la, lo, el = coords[0], coords[1], coords[2]        # 纬度，经度，高度
            for j_idx in range(len(channels)):
                j = channels[j_idx]
                j_ = j[:2]
                j_1, j_2, j_3 = j_ + "1", j_ + "2", j_ + "Z"
                j_e, j_n, j_z = j_ + "E", j_ + "N", j_ + "Z"
                if (j_1 in channels) & (j_2 in channels) & (j_3 in channels):
                    j_all = [j_1, j_2, j_3]
                    continue
                elif (j_e in channels) & (j_n in channels) & (j_z in channels):
                    j_all = [j_e, j_n, j_z]
                    continue
                else:
                    if j_idx == len(channels):
                        raise TypeError("Neither 12Z nor ENZ are available")
            j_ = j_all[0][:2]
            f.write(str(lo) + " " + str(la) + " " + net + " " + i + " " + j_ + " " + str(el))
            sta_idx = sta_idx + 1
            if sta_idx != num_sta:
                f.write("\n")           # 另起一行，统计下一个台站的信息
