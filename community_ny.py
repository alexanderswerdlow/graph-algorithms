# %%
import glob
import imageio
from datetime import datetime

import geopandas
import contextily as ctx
import community as community_louvain
import matplotlib.cm as cm
import networkx as nx
from collections import defaultdict
from pyproj import Transformer, CRS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

# %%
import geopandas as gpd
import shapefile
import pandas as pd

minn = CRS.from_proj4('+proj=lcc +lat_1=40.66666666666666 +lat_2=41.03333333333333 +lat_0=40.16666666666666 +lon_0=-74 +x_0=300000 +y_0=0 +datum=NAD83 +units=us-ft +no_defs')
crs_4326 = CRS.from_epsg(4326)
transformer = Transformer.from_crs(minn, crs_4326)

sf = shapefile.Reader("taxi_zones/taxi_zones.shp")
loc_map = {}
for row in sf.shapeRecords():
    print(row.record)
    x = (row.shape.bbox[0]+row.shape.bbox[2])/2
    y = (row.shape.bbox[1]+row.shape.bbox[3])/2
    pos = transformer.transform(x, y)
    loc_map[str(row.record[0])] = (pos[1], pos[0])

# %%
agg = pd.read_csv('yellow_tripdata_2020-01.csv', usecols=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'PULocationID', 'DOLocationID'])
rows = np.asarray(agg)

# %%
time_routes = defaultdict(int)
for st, et, dist, src, dst in rows:
    time_routes[(datetime.fromisoformat(st).weekday() + 1) * (datetime.fromisoformat(st).hour + 1)] += 1

# %%
for i in range(24):
    routes = defaultdict(int)
    for st, et, dist, src, dst in rows:
        if datetime.fromisoformat(st) == i:
            routes[tuple(np.sort([src, dst]))] += 1

    with open(f'ny_times/edges_{i}.txt', 'w') as f:
        for (src, dst), val in routes.items():
            if str(src) in loc_map and str(dst) in loc_map:
                f.write(f'{int(src)} {int(dst)} {val}\n')
            else:
                print(src, dst)

# # %%
# g = Graph.Read('taxi_edges.txt', format='ncol', directed=False)
# gcc = g.components().giant()

# # %%
# comms1 = gcc.community_multilevel(weights='weight')
# comms2 = gcc.community_fastgreedy(weights='weight')
# comms3 = gcc.community_edge_betweenness(weights='weight')
# #layout = g.layout("kk")
# plot(comms1, target="community1.png", mark_groups = True, vertex_size=4, edge_width=1)
# plot(comms2, target="community2.png", vertex_size=4, edge_width=1)
# plot(comms3, target="community3.png", vertex_size=4, edge_width=1)

# %%


def normalize(lst):
    s = sum(lst)
    return list(map(lambda x: 200 * (float(x)/s), lst))


for i in range(24):
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 2)

    ax3 = fig.add_subplot(gs[1:, :])
    G = nx.read_weighted_edgelist(f'ny_times/edges_{i}.txt')
    partition = community_louvain.community_louvain.best_partition(G)
    boros = geopandas.GeoDataFrame.from_file('taxi_zones/taxi_zones.shp').to_crs(4326)
    ax = boros.plot(ax=ax3, alpha=0.5,edgecolor='k')
    values = normalize([G[u][v]['weight'] for u, v in G.edges()])
    nx.draw(G, pos=loc_map,  ax=ax, node_color=list(partition.values()), node_size=5, width=values)
    #ctx.add_basemap(ax, crs=boros.crs.to_string())
    # source=ctx.providers.MapBox(accessToken = 'pk.eyJ1IjoiYXN3ZXJkbG93IiwiYSI6ImNrcGhrbTgwMjBvYzEycW13ZjIwNzg3ZTIifQ.Md-VXxUMRfXM9hImShPXXw')
    ax3.title.set_text(f"Hour: {i}")

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    degrees = [G.degree(n) for n in G.nodes()]
    ax1.hist(degrees, bins=25)
    ax1.title.set_text("Degree Distribution")
    ax1.set_xlabel('In Degree')
    ax1.set_ylabel('Frequency')

    degrees = np.asarray([G[u][v]['weight'] for u, v in G.edges()])
    edges = np.histogram_bin_edges(degrees)
    ax2.hist(degrees, bins=50)
    ax2.title.set_text("Edge weight Distribution")
    ax2.set_xlabel('Weight')
    ax2.set_ylabel('Frequency')
    plt.savefig(f'ny_times/community_{i}.png', dpi=300, bbox_inches='tight')
    plt.close()


# %%
images = []
with imageio.get_writer('ny_times/movie.gif', mode='I', duration=0.5) as writer:
    for filename in sorted(glob.glob('ny_times/community_*.png'), key=lambda x: int(x[:-4].split('_')[-1])):
        image = imageio.imread(filename)
        writer.append_data(image)


# import cv2
# import os
# image_folder = 'ny_times'
# images = sorted(glob.glob('ny_times/community_*.png'), key=lambda x: int(x[:-4].split('_')[-1]))
# frame = cv2.imread(images[0])
# height, width, layers = frame.shape
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# video = cv2.VideoWriter('ny_times/community_movie.mp4', fourcc, 1, (width,height))

# for image in images:
#     video.write(cv2.imread(image))

# cv2.destroyAllWindows()
# video.release()

#%%

mmin, mmax = (-9.95, 11.0)
for i in range(24):
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 2)
    G = nx.read_weighted_edgelist(f'ny_times/edges_{i}.txt', create_using=nx.DiGraph())

    boros = geopandas.GeoDataFrame.from_file('taxi_zones/taxi_zones.shp').to_crs(4326)


    for n, nbrsdict in G.adjacency():
        for nbr, eattr in nbrsdict.items():
            if "weight" in eattr:
                G.nodes[n]['flow'] = -eattr['weight'] + (G.nodes[n]['flow'] if 'flow' in G.nodes[n] else 0)
                G.nodes[nbr]['flow'] = eattr['weight'] + (G.nodes[nbr]['flow'] if 'flow' in G.nodes[nbr] else 0)


    color_map = []
    for n, nbrsdict in G.adjacency():
        color_map.append((n, np.log(G.nodes[n]['flow']) if np.log(G.nodes[n]['flow']) > 0 else -np.log(np.abs(G.nodes[n]['flow']))))

    d = {tup[0]: tup[1] for tup in color_map}

    color_map = []
    for j in range(263):
        if str(j) in d:
            color_map.append(d[str(j)])
        else:
            color_map.append(0)
    
    ax3 = fig.add_subplot(gs[1:, :])
    color_map = [(float(i)-mmin)/(mmax-mmin) for i in color_map]
    boros['color'] = [plt.cm.hot_r(x) for x in color_map]
    ax = boros.plot(alpha=0.5, edgecolor='k', color=boros['color'], ax=ax3)

    values = normalize([G[u][v]['weight'] for u, v in G.edges()])
    #nx.draw(G, pos=loc_map,  ax=ax, node_size=5, width=values)
    #ctx.add_basemap(ax, crs=boros.crs.to_string())
    # source=ctx.providers.MapBox(accessToken = 'pk.eyJ1IjoiYXN3ZXJkbG93IiwiYSI6ImNrcGhrbTgwMjBvYzEycW13ZjIwNzg3ZTIifQ.Md-VXxUMRfXM9hImShPXXw')
    ax3.title.set_text(f"Hour: {i}")

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    degrees = [G.in_degree(n) for n in G.nodes()]
    ax1.hist(degrees, bins=25)
    ax1.set_xlabel('In Degree')
    ax1.set_ylabel('Frequency')
    ax1.title.set_text("In Degree Distribution")

    degrees = [G.out_degree(n) for n in G.nodes()]
    ax2.hist(degrees, bins=25)
    ax2.set_xlabel('In Degree')
    ax2.set_ylabel('Frequency')
    ax2.title.set_text("Out Degree Distribution")

    plt.savefig(f'ny_times/heatmap_{i}.png', dpi=300, bbox_inches='tight')
    plt.close()

# %%
images = []
with imageio.get_writer('ny_times/heatmap.gif', mode='I', duration=0.5) as writer:
    for filename in sorted(glob.glob('ny_times/heatmap_*.png'), key=lambda x: int(x[:-4].split('_')[-1])):
        image = imageio.imread(filename)
        writer.append_data(image)




# %%
time_routes = defaultdict(int)
for st, et, dist, src, dst in rows:
    day = (datetime.fromisoformat(st).day)
    hour = (datetime.fromisoformat(st).hour + 1)
    groupp = np.digitize(datetime.fromisoformat(st).minute, [0, 15, 30, 45, 60])
    time_routes[day * hour * groupp] += 1

# %%
lists = sorted(time_routes.items())  # sorted by key, return a list of tuples
x, y = zip(*lists)  # unpack a list of pairs into two tuples
plt.plot(x, y)
plt.show()


# # calculate node betweenness centrality, weighted by travel time
# bc = nx.betweenness_centrality(D, weight="travel_time", normalized=True)
# nx.set_node_attributes(G, values=bc, name="bc")

# # plot the graph, coloring nodes by betweenness centrality
# nc = ox.plot.get_node_colors_by_attr(G, "bc", cmap="plasma")
# fig, ax = ox.plot_graph(
#     G, bgcolor="k", node_color=nc, node_size=50, edge_linewidth=2, edge_color="#333333"
# )


#%%
#from networkx import algorithms
G = nx.read_weighted_edgelist(f'taxi_edges.txt', create_using=nx.DiGraph())
for n, nbrsdict in G.adjacency():
    for nbr, eattr in nbrsdict.items():
        if "weight" in eattr:
            G.nodes[n]['flow'] = -eattr['weight'] + (G.nodes[n]['flow'] if 'flow' in G.nodes[n] else 0)
            G.nodes[nbr]['flow'] = eattr['weight'] + (G.nodes[nbr]['flow'] if 'flow' in G.nodes[nbr] else 0)

color_map = []
for n, nbrsdict in G.adjacency():
    color_map.append(G.nodes[n]['flow'])


plt.hist(color_map, density=True, bins=40)
plt.title('Probability Density Function of node net flow')
plt.xlabel('Net flow')
plt.ylabel('Probability')
plt.savefig('ny_times/net_flow_all.png', dpi=300, bbox='tight')
plt.show()
# algorithms.edge_betweenness(G)

# %%

routes = defaultdict(int)
for st, et, dist, src, dst in rows:
    if 5 <= datetime.fromisoformat(st).hour < 12:
        routes[tuple(np.sort([src, dst]))] += 1

with open(f'ny_times/taxi_edges_morning.txt', 'w') as f:
    for (src, dst), val in routes.items():
        if str(src) in loc_map and str(dst) in loc_map:
            f.write(f'{int(src)} {int(dst)} {val}\n')
        else:
            print(src, dst)
            
routes = defaultdict(int)
for st, et, dist, src, dst in rows:
    if 14 <= datetime.fromisoformat(st).hour < 24 or datetime.fromisoformat(st).hour < 5:
        routes[tuple(np.sort([src, dst]))] += 1

with open(f'ny_times/taxi_edges_night.txt', 'w') as f:
    for (src, dst), val in routes.items():
        if str(src) in loc_map and str(dst) in loc_map:
            f.write(f'{int(src)} {int(dst)} {val}\n')
        else:
            print(src, dst)

# %%
G = nx.read_weighted_edgelist(f'ny_times/taxi_edges_morning.txt', create_using=nx.DiGraph())
for n, nbrsdict in G.adjacency():
    for nbr, eattr in nbrsdict.items():
        if "weight" in eattr:
            G.nodes[n]['flow'] = -eattr['weight'] + (G.nodes[n]['flow'] if 'flow' in G.nodes[n] else 0)
            G.nodes[nbr]['flow'] = eattr['weight'] + (G.nodes[nbr]['flow'] if 'flow' in G.nodes[nbr] else 0)

color_map = []
for n, nbrsdict in G.adjacency():
    color_map.append(G.nodes[n]['flow'])

plt.figure()
plt.hist(color_map, density=True, bins=40)
plt.title('Probability Density Function of node net flow (6AM - 12PM)')
plt.xlabel('Net flow')
plt.ylabel('Probability')
plt.savefig('ny_times/net_flow_morning.png', dpi=300, bbox_inches='tight')

# %%
G = nx.read_weighted_edgelist(f'ny_times/taxi_edges_night.txt', create_using=nx.DiGraph())
for n, nbrsdict in G.adjacency():
    for nbr, eattr in nbrsdict.items():
        if "weight" in eattr:
            G.nodes[n]['flow'] = -eattr['weight'] + (G.nodes[n]['flow'] if 'flow' in G.nodes[n] else 0)
            G.nodes[nbr]['flow'] = eattr['weight'] + (G.nodes[nbr]['flow'] if 'flow' in G.nodes[nbr] else 0)

color_map = []
for n, nbrsdict in G.adjacency():
    color_map.append(G.nodes[n]['flow'])

plt.figure()
plt.hist(color_map, density=True, bins=40)
plt.title('Probability Density Function of node net flow (3PM - 5AM)')
plt.xlabel('Net flow')
plt.ylabel('Probability')
plt.savefig('ny_times/net_flow_night.png', dpi=300, bbox_inches='tight')



# %%
routes = []
for st, et, dist, src, dst in rows:
    dtt = ((datetime.fromisoformat(et) - datetime.fromisoformat(st)).seconds) / 60
    if dtt < 120:
        routes.append(dtt)

plt.figure()
plt.hist(routes, density=True, bins=60)
plt.title('Probability Density Function of trip length')
plt.xlabel('Trip Length (Minutes)')
plt.ylabel('Probability')
plt.savefig('ny_times/trip_length.png', dpi=300, bbox_inches='tight')



#%% 
from networkx import algorithms
G = nx.read_weighted_edgelist(f'taxi_edges.txt' )
boros = geopandas.GeoDataFrame.from_file('taxi_zones/taxi_zones.shp').to_crs(4326)
d = algorithms.betweenness_centrality(G, weight='weight')

color_map = []
for j in range(263):
    if str(j) in d:
        color_map.append(d[str(j)])
    else:
        color_map.append(0)

color_map = [(float(i)-min(color_map))/(max(color_map)-min(color_map)) for i in color_map]
boros['color'] = [plt.cm.hot_r(x) for x in color_map]
boros.plot(alpha=0.5, edgecolor='k', legend=True, color=boros['color'], figsize=(9, 9))
plt.title('Shortest-Path Betweenness Centrality for Nodes')
plt.savefig('betweenness_undirected.png', dpi=300, bbox_inches='tight')
#values = normalize([G[u][v]['weight'] for u, v in G.edges()])
#nx.draw(G, pos=loc_map,  ax=ax, node_size=5, width=values)
#ctx.add_basemap(ax, crs=boros.crs.to_string())
# source=ctx.providers.MapBox(accessToken = 'pk.eyJ1IjoiYXN3ZXJkbG93IiwiYSI6ImNrcGhrbTgwMjBvYzEycW13ZjIwNzg3ZTIifQ.Md-VXxUMRfXM9hImShPXXw')
# %%
G = nx.read_weighted_edgelist(f'taxi_edges.txt')
boros = geopandas.GeoDataFrame.from_file('taxi_zones/taxi_zones.shp').to_crs(4326)
partition = community_louvain.community_louvain.best_partition(G)

defined_colors = ['#0053ed', '#00e047', '#00e047', '#e0008e', '#ffff2b', '#ffffff']
color_map = []
for j in range(263):
    if str(j) in partition:
        color_map.append(defined_colors[int(partition[str(j)])])
    else:
        color_map.append(defined_colors[5])

# color_map_ = [(float(i)-min(color_map))/(max(color_map)-min(color_map)) for i in color_map]


boros['color'] = color_map
boros.plot(alpha=0.5, edgecolor='k', legend=True, color=boros['color'], figsize=(9, 9))
plt.title('Louvain Community Structure (All Time)')
plt.savefig('structure_map.png', dpi=300, bbox_inches='tight')
# %%


#%%
time_routes = defaultdict(int)
for st, et, dist, src, dst in rows:
    time_routes[(datetime.fromisoformat(st).weekday() + 1) * (datetime.fromisoformat(st).hour + 1)] += 1

#%%
dff = []
for k in range(1, 169):
    dff.append(time_routes[k])

#%%
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
#
# Run the test
#
df_stationarityTest = adfuller(dff, autolag='AIC')
#
# Check the value of p-value
#
print("P-value: ", df_stationarityTest[1])
#
# Next step is to find the order of AR model to be trained
# for this, we will plot partial autocorrelation plot to assess
# the direct effect of past data on future data
#
from statsmodels.graphics.tsaplots import plot_pacf
pacf = plot_pacf(dff, lags=25)