#%%
from geopy.geocoders import Nominatim, MapBox
import itertools

import json
import random
import sys
from collections import defaultdict
from itertools import tee
from math import asin, cos, radians, sin, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from igraph import *
from scipy.spatial import Delaunay
import matplotlib
import networkx as nx
random.seed(0)

# Taken from haversine package


def haversine(point1, point2):
    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1 = radians(lat1)
    lng1 = radians(lng1)
    lat2 = radians(lat2)
    lng2 = radians(lng2)

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lng * 0.5) ** 2

    avg_earth_radius_miles = (6371.0088 * 0.621371192)
    return 2 * avg_earth_radius_miles * asin(sqrt(d))



agg = pd.read_csv('los_angeles-censustracts-2019-4-All-MonthlyAggregate.csv', usecols=['sourceid', 'dstid', 'mean_travel_time', 'month'])
agg = agg.drop(agg[agg.month != 12].index)
agg = agg.drop(['month'], axis=1)

rows = np.asarray(agg)

route_edges = []
routes = defaultdict(list)
for src, dst, mean_time in rows:
    routes[tuple(np.sort([src, dst]))].append(mean_time)

for (src, dst), mean_time in routes.items():
    route_edges.append((src, dst, np.mean(mean_time)))

g = Graph.TupleList(route_edges, weights=True, directed=False)
gcc = g.components().giant()
#gcc=g
print(f'Original Graph — Vertices: {len(g.vs)} Edges: {len(g.es)}')
print(f'Cleaned Graph — Vertices: {len(gcc.vs)} Edges: {len(gcc.es)}')

#%%
# Q6

location_data = {}
with open('los_angeles_censustracts.json', 'r') as f:
    for feature in json.loads(f.readline())['features']:
        coordinates = np.array(feature['geometry']['coordinates'][0][0])
        mean_location = None
        if coordinates.ndim == 1:
            mean_location = np.mean(coordinates.reshape(1, -2), axis=0)
        else:
            mean_location = np.mean(coordinates, axis=0)

        location_data[feature['properties']['MOVEMENT_ID']] = {
            'address': feature['properties']['DISPLAY_NAME'],
            'location': mean_location}

for v in gcc.vs():
    v['address'] = location_data[np.str(np.int(v['name']))]['address']
    v['location'] = location_data[np.str(np.int(v['name']))]['location']

# %% Q7
mst = gcc.spanning_tree(weights=gcc.es["weight"])
print(f'Original Graph — Vertices: {len(mst.vs)} Edges: {len(mst.es)}')

# Sample 5 edges
locs = np.empty((1, 2))
for idx, i in enumerate(random.sample(range(1, len(mst.es()) - 1), 5)):
    src, dst = mst.vs()[mst.es()[i].source], mst.vs()[mst.es()[i].target]
    print(src['location'])

    geolocator = Nominatim(user_agent="oiuqwjinkhz")
    src_adr, dst_adr = geolocator.reverse(f"{src['location'][1]}, {src['location'][0]}").address, geolocator.reverse(f"{dst['location'][1]}, {dst['location'][0]}").address

    # Mapbox
    # geolocator = MapBox(api_key='pk.eyJ1IjoiYXN3ZXJkbG93IiwiYSI6ImNrcGhrbTgwMjBvYzEycW13ZjIwNzg3ZTIifQ.Md-VXxUMRfXM9hImShPXXw')
    # src_adr, dst_adr = geolocator.reverse(np.flip(src['location'])).address, geolocator.reverse(np.flip(dst['location'])).address
    # fig.update_layout(
    #     mapbox=dict(
    #         accesstoken='pk.eyJ1IjoiYXN3ZXJkbG93IiwiYSI6ImNrcGhrbTgwMjBvYzEycW13ZjIwNzg3ZTIifQ.Md-VXxUMRfXM9hImShPXXw',
    #         zoom=13
    #     ),
    # )

    locs = np.vstack((src['location'], dst['location']))
    fig = px.line_mapbox(locs, lat=1, lon=0, zoom=14)

    print(f"Src: {src_adr}\nDst: {src_adr})\nSrc Name: {src['address']} Dst Name: {dst['address']}")
    print(f"Src Coord: {src['location']} Dst Coords: {dst['location']}, Dist: {haversine(src['location'], dst['location'])}")

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    fig.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[
            {
                "below": 'traces',
                "sourcetype": "raster",
                "sourceattribution": "United States Geological Survey",
                "source": [
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                ]
            }
        ])

    fig.show()
    fig.write_image(f'Q7-{idx}.png')

# %% Q8


def valid(e): return gcc.get_eid(e[0], e[1], directed=False, error=False) != -1
def get_edge(e): return gcc.get_eid(e[0], e[1], directed=False, error=False)


num_invalid = 0
for i in range(1000):
    vertices = random.sample(range(1, len(mst.vs()) - 1), 3)

    # Make sure the vertices are all connected with an edge
    while not (valid(vertices[:2]) and valid(vertices[1:]) and valid(vertices[::2])):
        vertices = random.sample(range(1, len(mst.vs()) - 1), 3)

    weights = [gcc.es(get_edge(e))['weight'][0] for e in itertools.combinations(vertices, 2)]

    for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        if weights[i] + weights[j] < weights[k]:
            num_invalid += 1
            break

print(f'Percent that satisfy triangle inequality: {1 - num_invalid / 1000}')

# %% Q9

# Taken from https://github.com/feynmanliang/Euler-Tour


def find_euler_tour(graph):
    tour = []
    E = graph

    numEdges = defaultdict(int)

    def find_tour(u):
        for e in range(len(E)):
            if E[e] == 0:
                continue
            if u == E[e][0]:
                u, v = E[e]
                E[e] = 0
                find_tour(v)
            elif u == E[e][1]:
                v, u = E[e]
                E[e] = 0
                find_tour(v)
        tour.insert(0, u)

    for i, j in graph:
        numEdges[i] += 1
        numEdges[j] += 1

    start = graph[0][0]
    for i, j in numEdges.items():
        if j % 2 > 0:
            start = i
            break

    current = start
    find_tour(current)

    if tour[0] != tour[-1]:
        return None
    return tour


sys.setrecursionlimit(100000)
edges = [e.tuple for e in mst.es()]
multi_graph = mst.as_undirected()
multi_graph.add_edges(edges)
graph = [e.tuple for e in multi_graph.es()]
cycle = find_euler_tour(graph)
tour = list(dict.fromkeys(cycle))

# %%


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


approx_path = []
for v_1, v_2 in pairwise(tour):
    edges = gcc.es(_between=([v_1], [v_2]))
    if len(edges):
        v_1_g = gcc.vs(name_eq=mst.vs(v_1)[0]['name'])
        approx_path.append((v_1_g, edges[0]['weight']))
    else:
        inner_path = gcc.get_shortest_paths(v_1, v_2, weights=gcc.es()['weight'])[0]
        i_weight = 0
        for k_1, k_2 in pairwise(inner_path):
            e = gcc.es.select(_between=([k_1], [k_2]))[0]
            approx_path.append((gcc.vs()[k_1], e['weight']))
            i_weight += e['weight']

approx_path_weight = sum(n for _, n in approx_path)
print(f'Approx path weight: {approx_path_weight}')
print(approx_path_weight / sum(mst.es['weight']))

# %% Q10

locations = np.stack([location_data[v['name'][0]]['location'] for v, _ in approx_path], axis=0)
plt.figure(figsize=(25, 20))
plt.plot(locations[:, 0], locations[:, 1])
plt.title("Approximate TSP Tour")
plt.xlabel('Longitude (Deg)')
plt.ylabel('Latitude (Deg)')
plt.savefig('Q10.png', dpi=300, bbox_inches='tight')
plt.show()

fig = px.line_mapbox(locations, lat=1, lon=0)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_layout(
    mapbox=dict(
        accesstoken='pk.eyJ1IjoiYXN3ZXJkbG93IiwiYSI6ImNrcGhrbTgwMjBvYzEycW13ZjIwNzg3ZTIifQ.Md-VXxUMRfXM9hImShPXXw',
        zoom=8.5
    ),
)
fig.show()
fig.write_image('Q10_1.png')

# %% Q11

locs = np.empty((len(gcc.vs()), 2))
locs[:] = gcc.vs()[:]['location']
tri = Delaunay(locs)

plt.triplot(locs[:, 0], locs[:, 1], tri.simplices)
plt.plot(locs[:, 0], locs[:, 1], 'r.')
plt.title("Road Mesh")
plt.xlabel('Longitude (Deg)')
plt.ylabel('Latitude (Deg)')
plt.savefig('Q11.png', dpi=300, bbox_inches='tight')
plt.show()


fig = px.line_mapbox(locs, lat=1, lon=0)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_layout(
    mapbox=dict(
        accesstoken='pk.eyJ1IjoiYXN3ZXJkbG93IiwiYSI6ImNrcGhrbTgwMjBvYzEycW13ZjIwNzg3ZTIifQ.Md-VXxUMRfXM9hImShPXXw',
        zoom=8.5
    ),
)
fig.show()
fig.write_image('Q11_1.png')

# %% Q13
tri_edges = []
vertex_array = []
all_edges = []
for triangle in tri.simplices:
    for i, j in ((0, 1), (1, 2), (0, 2)):
        v_1, v_2 = gcc.vs()[triangle[i]]['location'], gcc.vs()[triangle[j]]['location']
        #print(v_1, v_2)
        edge_length = 69 * np.linalg.norm(v_1 - v_2)  # Nice
        edge = gcc.es(_between=([triangle[i]], [triangle[j]]))
        if len(edge) > 0:
            speed = edge_length / (edge[0]['weight']/3600)
            flow = 2*speed / (0.003 + speed/1800)  # 2 at nomenator - 2 lines; edge_length/edge[0]['weight']*2 - distcance between cars
            if flow > 0 and [triangle[i], triangle[j]] not in vertex_array and [triangle[j], triangle[i]] not in vertex_array:
                #tri_edges.append((gcc.vs()[triangle[i]]['name'], gcc.vs()[triangle[i]]['name'], edge[0]['weight'], flow))
                tri_edges.append((gcc.vs()[triangle[i]]['name'], gcc.vs()[triangle[j]]['name'], edge[0]['weight'], flow))
                vertex_array.append([triangle[i], triangle[j]])
                
                all_edges.append(edge[0]['weight'])

g = Graph.TupleList(tri_edges, edge_attrs=['weight', 'capacity'])

for v in g.vs():
    v['location'] = gcc.vs(name_eq=v['name'])[0]['location']

# %% Q13-2

# TODO: Currently max flow is broken
src = g.vs.select(lambda v: np.isclose(v['location'], np.array([-118.56, 34.04]), atol=2.5e-2).all())
dst = g.vs.select(lambda v: np.isclose(v['location'], np.array([-118.18, 33.77]), atol=1e-3).all())

max_flow = g.maxflow(src[0].index, dst[0].index, capacity='capacity')
edge_disjoint_paths = g.maxflow_value(src[0].index, dst[0].index)

print(f'Max Traffic flow from malibu to long beach: {max_flow.value}')
print(f'Edge disjoint paths between malibu and long beach: {edge_disjoint_paths}')



#%% Q14
matplotlib.use('TkAgg') 
all_weights = gcc.es[:]['weight']

thresh = np.mean(all_edges) + 3*np.std(all_edges)


a = np.histogram(all_edges, bins=100)
plt.hist(a, bins = 100)
plt.savefig('Q13_1.png')

tri_edges = []
vertex_array = []

nedges = 0
for triangle in tri.simplices:
    for i, j in ((0, 1), (1, 2), (0, 2)):
        v_1, v_2 = gcc.vs()[triangle[i]]['location'], gcc.vs()[triangle[j]]['location']
        #print(v_1, v_2)
        edge_length = 69 * np.linalg.norm(v_1 - v_2)  # Nice
        edge = gcc.es(_between=([triangle[i]], [triangle[j]]))
        if len(edge) > 0:
            speed = edge_length / (edge[0]['weight']/3600)
            flow = 2*speed / (0.003 + speed/1800)  # 2 at nomenator - 2 lines; edge_length/edge[0]['weight']*2 - distcance between cars
            if flow > 0 and [triangle[i], triangle[j]] not in vertex_array and [triangle[j], triangle[i]] not in vertex_array and edge[0]['weight'] < 100:
                #tri_edges.append((gcc.vs()[triangle[i]]['name'], gcc.vs()[triangle[i]]['name'], edge[0]['weight'], flow))
                tri_edges.append((gcc.vs()[triangle[i]]['name'], gcc.vs()[triangle[j]]['name'], edge[0]['weight'], flow))
                vertex_array.append([triangle[i], triangle[j]])
                nedges += 1
                
g = Graph.TupleList(tri_edges, edge_attrs=['weight', 'capacity'])

for v in g.vs():
    v['location'] = gcc.vs(name_eq=v['name'])[0]['location']



#%% Q15
src = g.vs.select(lambda v: np.isclose(v['location'], np.array([-118.56, 34.04]), atol=2.5e-2).all())
dst = g.vs.select(lambda v: np.isclose(v['location'], np.array([-118.18, 33.77]), atol=1e-3).all())

max_flow = g.maxflow(src[0].index, dst[0].index, capacity='capacity')
edge_disjoint_paths = g.maxflow_value(src[0].index, dst[0].index)

print(f'Max Traffic flow from malibu to long beach: {max_flow.value}')
print(f'Edge disjoint paths between malibu and long beach: {edge_disjoint_paths}')

locs =  np.asarray(g.vs[:]['location'])  
fig = px.line_mapbox(locs, lat=1, lon=0)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_layout(
    mapbox=dict(
        accesstoken='pk.eyJ1IjoiYXN3ZXJkbG93IiwiYSI6ImNrcGhrbTgwMjBvYzEycW13ZjIwNzg3ZTIifQ.Md-VXxUMRfXM9hImShPXXw',
        zoom=8.5
    ),
)
fig.show()
fig.write_image('Q13_2.png')


