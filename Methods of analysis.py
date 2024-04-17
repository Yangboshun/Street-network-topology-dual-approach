import osmnx as ox
import momepy
import geopandas as gpd
import matplotlib.pyplot as plt
from libpysal import weights
import collections
#原始图
streets_graph = ox.graph_from_place('Manhattan, USA', network_type='drive')
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

#新道路与原始道路的对应关系
stroke_attr = coins.stroke_attribute()
stroke_attr.plot(markersize=0.01, linewidth=0.5)
stroke_attr

W = weights.Queen.from_dataframe(stroke_gdf)
# Convert the graph to networkx
G_dual = W.to_networkx()

import networkx as nx
# 创建一个绘图对象
fig, ax = plt.subplots(figsize=(10, 10))

# 绘制双重图
nx.draw(G_dual, ax=ax, node_size=10, node_color='b', edge_color='gray')

# 显示图形
plt.show()

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import Counter
#标准
degree_sequence = [d for n, d in G_dual.degree()]
degree_count = Counter(degree_sequence)
deg, cnt = zip(*degree_count.items())

# Create a scatter plot
scatter = go.Scatter(
    x=deg,
    y=cnt,
    mode='markers',
    marker=dict(size=10, color='blue'),
    text=deg,
    hoverinfo='text+y'
)

# Create layout
layout = go.Layout(
    title="Kangding",
    xaxis=dict(title='Degree'),
    yaxis=dict(title='Frequency'),
    hovermode='closest'
)

# Create the figure
fig = go.Figure(data=[scatter], layout=layout)

# Show the figure using Plotly
fig.show()

# Show the figure using Matplotlib
plt.scatter(deg, cnt)
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution')
plt.show()

#正态性
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy.stats import lognorm
from scipy.optimize import curve_fit

degree_sequence = [d for n, d in G_dual.degree()]  # use raw degree, not sorted
degree_count = Counter(degree_sequence)
deg, cnt = zip(*degree_count.items())

# 将计数(cnt)转换为概率(pk)
total = sum(cnt)
pk = [c / total for c in cnt]

# 拟合函数使用log-normal的概率密度函数
def log_normal(x, mu, sigma):
    return (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))

# 对原始序列拟合使用log-normal分布
params, _ = curve_fit(log_normal, deg, pk, p0=[1,1])

# 根据拟合参数生成数据
xdata = np.linspace(min(deg), max(deg), 1000)
pdf_fitted = log_normal(xdata, *params)

# 准备图表
fig, ax = plt.subplots()
ax.scatter(deg, pk, marker='o')  # 现在是按概率绘制散点图

# 绘制拟合的log-normal分布曲线
ax.plot(xdata, pdf_fitted, 'r', label='Lognormal fit')

# 设置横纵坐标的比例
ax.set_xscale('log')
ax.set_yscale('linear')  # P(k) 是个概率值，应该使用线性规模

# 标记坐标轴和标题
plt.title('Degree Distribution with Lognormal Fit')
plt.ylabel('P(k)')
plt.xlabel('k ')

# 加网格
plt.grid(True, which='both', linestyle='--', linewidth='0.5')

# 添加图例
plt.legend()

plt.show()

#log-log
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
from collections import Counter

degree_sequence = sorted([d for n, d in G_dual.degree()], reverse=True)  # Sorted Degree sequence

# Fit the power-law distribution to the degree sequence
results = powerlaw.Fit(degree_sequence)
gamma = results.power_law.alpha
xmin = results.power_law.xmin

# Create a Counter of the degree distribution
degreeCount = Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

# Convert frequency to probability
total_nodes = sum(cnt)
prob = [c / total_nodes for c in cnt]

# Store the minimum probability before applying the powerlaw fit for the plot y-limits
min_prob_before_fit = min(prob)

# Prepare the figure
fig, ax = plt.subplots()

# Plot the original degree distribution with probabilities
ax.scatter(deg, prob, marker='o', label='Empirical Data')  # Using scatter for individual data points

# Set scale of axes to log
ax.set_xscale('log')
ax.set_yscale('log')

# Set the ylim to accommodate the minimum data probability before the fit
ax.set_ylim(bottom=min_prob_before_fit)

# Label the axes and title the plot
ax.set_title('Degree Distribution (Log-Log scale)')
ax.set_xlabel('k')
ax.set_ylabel('P(k)')

# Overlay the power-law distribution fit (in red)
results.power_law.plot_pdf(color='r', linestyle='--', label=f'Power-law fit with γ = {gamma:.3f}')

# Add the legend, which includes the gamma value
ax.legend()

# Add grid for better readability
ax.grid(True, which='both', linestyle='--', linewidth='0.5')

# Display the plot
plt.show()

#仿真小世界系数
import networkx as nx

# 计算实际网络的聚类系数和平均最短路径长度
C_real = nx.average_clustering(G_dual)
L_real = nx.average_shortest_path_length(G_dual)

# 生成随机网络
# 假设每个节点的连接概率p是通过实际网络的平均度数除以节点数-1得到的
p = average_degree / (len(nodes) - 1)
G_random = nx.erdos_renyi_graph(len(nodes), p)

# 计算随机网络的聚类系数和平均最短路径长度
C_random = nx.average_clustering(G_random)
L_random = nx.average_shortest_path_length(G_random)

# 计算小世界系数
sigma = (C_real / C_random) / (L_real / L_random)

# 打印小世界系数
print("小世界系数:", sigma)
print(p)