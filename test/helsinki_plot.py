import pyrosm 
from pyrosm import get_data
from matplotlib import pyplot as plt

fp = get_data('helsinki')
osm = pyrosm.OSM(fp)
drive_net = osm.get_buildings()

drive_net.plot(figsize=(200, 200))
plt.savefig('helsinki_roads.png', dpi=500)