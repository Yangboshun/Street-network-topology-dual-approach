#实验-小型案例测试
import osmnx as ox
import momepy
import geopandas as gpd
import matplotlib.pyplot as plt
from libpysal import weights
import collections
import matplotlib.cm as cm
import networkx as nx
import numpy as np

#原始图的拓扑构建
#设置检索坐标与范围
wurster_hall = (41.4071495, 2.1632748)
one_mile = 200 #米
graph = ox.graph_from_point(wurster_hall, dist=one_mile, network_type="drive")
fig, ax = ox.plot_graph(graph, figsize=(8, 8), node_size=10, node_color="hotpink", edge_color="gray", edge_linewidth=0.5, bgcolor='white')

#构建为无向图
M = ox.utils_graph.get_undirected(graph)

#投影
G_proj = ox.project_graph(M)
nodes_proj = ox.graph_to_gdfs(G_proj, edges=False)
graph_area_m = nodes_proj.unary_union.convex_hull.area
graph_area_m

#拓扑网络基本信息
ox.basic_stats(G_proj, area=graph_area_m, clean_int_tol=15)

#对偶图构建，代码略有差异
wurster_hall = (41.4071495, 2.1632748)
one_mile = 200
streets_graph = ox.graph_from_point(wurster_hall, dist=one_mile, network_type="drive")
streets = ox.graph_to_gdfs(ox.get_undirected(streets_graph), nodes=False, edges=True,
                                   node_geometry=False, fill_edge_geometry=True)

W = weights.Queen.from_dataframe(streets)
G_dual = W.to_networkx()

# 选择合适的layout
pos = nx.circular_layout(G_dual)

# 可视化调整
degree_sequence = [d for n, d in G_dual.degree()]
max_degree = max(degree_sequence)
min_degree = min(degree_sequence)
degree_color = cm.plasma([(d - min_degree) / (max_degree - min_degree) for d in degree_sequence])

#减小节点大小并添加透明度(alpha)
nx.draw(G_dual, pos, node_size=15, node_color=degree_color, edge_color='gray', width=0.5, alpha=0.5)

# 可视化
plt.show()

# 获取节点连边
edges = G_dual.edges()
nodes = G_dual.nodes()

# 计算平均度
average_degree = sum(dict(G_dual.degree()).values()) / len(nodes)

# 计算集聚系数
clustering_coefficient = nx.average_clustering(G_dual)

# 计算网络直径
diameter = nx.diameter(G_dual)

# 计算平均最短路径长度
average_shortest_path_length = nx.average_shortest_path_length(G_dual)

# 计算节点中心性
degree_centrality = nx.degree_centrality(G_dual)
closeness_centrality = nx.closeness_centrality(G_dual)
betweenness_centrality = nx.betweenness_centrality(G_dual)

# 打印网络信息
print("节点数量:", len(nodes))
print("连边数量:", len(edges))
print("平均度:", average_degree)
print("集聚系数:", clustering_coefficient)
print("网络直径:", diameter)
print("平均最短路径长度:", average_shortest_path_length)
print("节点度中心性:", degree_centrality)
print("节点接近中心性:", closeness_centrality)
print("节点介数中心性:", betweenness_centrality)

#使用Every-best-fit进行拓扑重构，并进行对偶图构建
wurster_hall = (41.4071495, 2.1632748)
one_mile = 200
streets_graph = ox.graph_from_point(wurster_hall, dist=one_mile, network_type="drive")
streets = ox.graph_to_gdfs(ox.get_undirected(streets_graph), nodes=False, edges=True,
                                   node_geometry=False, fill_edge_geometry=True)

#coins算法合并道路段
coins = momepy.COINS(streets)
stroke_gdf = coins.stroke_gdf()
stroke_gdf.plot(markersize=0.01, linewidth=0.5)
stroke_gdf

stroke_gdf.plot(cmap="tab10", linewidth=0.7, figsize=(10, 10)).set_axis_off()

W = weights.Queen.from_dataframe(stroke_gdf)

G_dual = W.to_networkx()

pos = nx.circular_layout(G_dual)
degree_sequence = [d for n, d in G_dual.degree()]
max_degree = max(degree_sequence)
min_degree = min(degree_sequence)
degree_color = cm.plasma([(d - min_degree) / (max_degree - min_degree) for d in degree_sequence])
nx.draw(G_dual, pos, node_size=15, node_color=degree_color, edge_color='gray', width=0.5, alpha=0.5)
plt.show()

#现实网络案例-河北省衡水市桃城区-作者的家乡
import osmnx as ox
import momepy
import geopandas as gpd
import matplotlib.pyplot as plt
from libpysal import weights
import collections

streets_graph = ox.graph_from_place('桃城区，衡水，中国', network_type='drive')
streets = ox.graph_to_gdfs(ox.get_undirected(streets_graph), nodes=False, edges=True,
                                   node_geometry=False, fill_edge_geometry=True)

streets.plot(figsize=(10, 10), linewidth=0.2).set_axis_off()

#查看连边属性
streets.reset_index(inplace=True)
streets.plot(markersize=0.01, linewidth=0.5)
streets

#coins算法合并道路段
coins = momepy.COINS(streets)
stroke_gdf = coins.stroke_gdf()
stroke_gdf.plot(markersize=0.01, linewidth=0.5)
stroke_gdf

#以naturalbreaks方法，基于街道长度进行分类
stroke_gdf.plot(stroke_gdf.length,
                figsize=(15, 15),
                cmap="copper_r",
                linewidth=0.7,
                scheme="naturalbreaks", k=4
               ).set_axis_off()

#对偶
W = weights.Queen.from_dataframe(stroke_gdf)

G_dual = W.to_networkx()
plt.show()

import matplotlib.cm as cm
import networkx as nx
import numpy as np

pos = nx.kamada_kawai_layout(G_dual)
degree_sequence = [d for n, d in G_dual.degree()]
max_degree = max(degree_sequence)
min_degree = min(degree_sequence)
degree_color = cm.plasma([(d - min_degree) / (max_degree - min_degree) for d in degree_sequence])
nx.draw(G_dual, pos, node_size=5, node_color=degree_color, edge_color='gray', width=0.5, alpha=0.5)
plt.show()

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import Counter

# 基本的度分布绘制-可交互
degree_sequence = [d for n, d in G_dual.degree()]
degree_count = Counter(degree_sequence)
deg, cnt = zip(*degree_count.items())

# 散点图
scatter = go.Scatter(
    x=deg,
    y=cnt,
    mode='markers',
    marker=dict(size=10, color='blue'),
    text=deg,
    hoverinfo='text+y'
)

# 创建layout
layout = go.Layout(
    title="Kangding",
    xaxis=dict(title='Degree'),
    yaxis=dict(title='Frequency'),
    hovermode='closest'
)

fig = go.Figure(data=[scatter], layout=layout)
fig.show()

# 非可交互
plt.scatter(deg, cnt)
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution')
plt.show()