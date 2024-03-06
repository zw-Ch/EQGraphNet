from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt


client = Client("GFZ")

t = UTCDateTime("2007/01/02 05:48:50")
st = client.get_waveforms(network="CX", station="PB01", location="*", channel="HH?", starttime=t-100, endtime=t+100)

print()
