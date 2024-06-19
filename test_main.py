import forsys as fsys
import os
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
import re
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objs as go

# import forsys.surface_evolver as surface_evolver
from forsys.surface_evolver import SurfaceEvolver

# How to plot 2D in 3D coordinates
# https://stackoverflow.com/questions/36470343/how-to-plot-2d-matplotlib-graphs-in-3d-space
# https://stackoverflow.com/questions/44881885/plotting-2d-plot-into-3d-plot-in-matplotlib
# how to deal with faces and pressures in 3D plotting
# what's the format and requirements of Augusto's code
# how to improve the plotting function

filepath = r"C:\Users\takeoff\Desktop\GRINDING\test_output0.dmp"  # read the file

# get the vertices, edges and cells
# def calc_index(filepath):
with open(os.path.join(filepath), "r") as f:
    lines = [next(f) for _ in range(5)]  # 读取前五行进行检查
    print(lines)  # 打印这些行以检查是否正确读取
    f.seek(0)
    ini_v = next(index for index, line in enumerate(f) if line.startswith("vertices  "))
    fin_v = next(index for index, line in enumerate(f) if line.startswith("edges  ")) + ini_v  # FINV = INIE + INIV
    if ini_v is None:
        raise ValueError("Line starts with 'vertices ' not found。")
    else:
        print(f"Vertices defined from  {ini_v} to {fin_v} 。")

    f.seek(0)
    ini_e = next(index for index, line in enumerate(f) if line.startswith("edges  "))
    fin_e = next(index for index, line in enumerate(f) if line.startswith("faces  ")) + ini_e  #
    if ini_e is None:
        raise ValueError("Line starts with 'edges ' not found。")
    else:
        print(f"Edges defined from  {ini_e} to {fin_e} 。")

    f.seek(0)
    ini_f = next(index for index, line in enumerate(f) if line.startswith("faces  "))
    fin_f = next(index for index, line in enumerate(f) if line.startswith("bodies  ")) + ini_f  #
    if ini_f is None:
        raise ValueError("Line starts with 'faces ' not found。")
    else:
        print(f"Faces defined from  {ini_f} to {fin_f} 。")

    f.seek(0)
    ini_p = next(index for index, line in enumerate(f) if line.startswith("bodies  "))
    fin_p = next(index for index, line in enumerate(f) if line.startswith("read")) + ini_p
    if ini_p is None:
        raise ValueError("Line starts with 'bodies ' not found。")
    else:
        print(f"Bodies defined from  {ini_p} to {fin_p} 。")

index_v = (ini_v, fin_v)
index_e = (ini_e, fin_e)
index_f = (ini_f, fin_f)
index_p = (ini_p, fin_p)
# print(index_v, index_e, index_f, index_p)
# return index_v, index_e, index_f, index_p

# calc_index(filepath)

# def get_edges(filepath):
edges = pd.DataFrame()
ids = []
id1 = []
id2 = []
forces = []

with open(os.path.join(filepath), "r") as f:
    lines = f.readlines()
    for i in range(index_e[0] + 1, index_e[1]):
        ids.append(int(re.search(r"\d+", lines[i]).group()))
        id1.append(int(lines[i].split()[1]))
        id2.append(int(lines[i].split()[2]))
        forces.append(float(lines[i].split()[4]) if lines[i].split()[3] == "density" else 1)
        # vals.append(float(lines[i][lines[i].find("density"):].split(" ")[1]))
edges['id'] = ids
edges['id1'] = id1
edges['id2'] = id2
edges['force'] = forces
# return edges

# def get_vertices(filepath):
vertices = pd.DataFrame()
ids = []
xs = []
ys = []

# index_v, _, _, _ = self.get_first_last()

with open(os.path.join(filepath), "r") as f:
    lines = f.readlines()
    for i in range(index_v[0] + 1, index_v[1]):
        ids.append(int(re.search(r"\d+", lines[i]).group()))
        xs.append(round(float(lines[i].split()[1]), 3))
        ys.append(round(float(lines[i].split()[2]), 3))
        # vals.append(float(lines[i][lines[i].find("density"):].split(" ")[1]))
vertices['id'] = ids
vertices['x'] = xs
vertices['y'] = ys

# def get_cells(filepath):
cells = pd.DataFrame()
ids = []
edges_list = []  # edges 被重新定义，这里导致后面的匹配出错

pressure_dict = {}
pressures = {}
# _, _, _, index_p = self.get_first_last()
# to get the pressures
with open(os.path.join(filepath), "r") as f:
    lines = f.readlines()
    for i in range(index_p[0] + 1, index_p[1]):
        splitted = lines[i].split() # split函数默认以空格分割，返回一个列表
        if len(splitted) > 0:
            pressures[int(splitted[0])] = float(splitted[7]) # 读取每一行的数据的第一个和第八个数据
        else:
            break
pressure_dict = pressures

# to get the cells
with open(os.path.join(filepath), "r") as f:
    lines = f.readlines()
    current_edge = []
    first = True

    for i in range(index_f[0] + 1, index_f[1]):
        splitted = lines[i].split()
        if first and "*/" not in splitted[-1]:
            ids.append(splitted[0])
            first = False
            current_edge = current_edge + splitted[1:-1]
        elif splitted[-1] == '\\' and not first:
            current_edge = current_edge + splitted[0:-1]
        elif "*/" in splitted[-1]:
            # last line, save
            if first:
                ids.append(splitted[0])
                current_edge = current_edge + splitted[1:-2]
            else:
                current_edge = current_edge + splitted[0:-2]
            edges_list.append([int(e) for e in current_edge])
            current_edge = []
            first = True

cells['id'] = ids
# print(edges_list)

cells['edges'] = edges_list

cells["pressures"] = pressure_dict.values()

# print(cells.columns)


print(vertices['id'])
# frame = SurfaceEvolver(filepath)

# now use plot.py to plot with vertices, edges and cells


def calculate_centroid(vertex_ids):
    # 从 vertices DataFrame 中筛选出当前单元格的顶点
    vertex_subset = vertices[vertices['id'].isin(vertex_ids)]
    # 计算这些顶点的x和y坐标的平均值，得到质心坐标
    centroid_x = np.mean(vertex_subset['x'])
    centroid_y = np.mean(vertex_subset['y'])
    return centroid_x, centroid_y


def plot_mesh_v3(vertices, edges, cells, name="0", folder=".", xlim=[], ylim=[], zlim=[], mirror_y=False,
                  mirror_x=False):
    fig, ax = plt.subplots(1, 1)
    if not os.path.exists(folder):
        os.makedirs(folder)
    to_save = os.path.join(folder, name)

    for i, v in vertices.iterrows():
        plt.scatter(v['x'], v['y'], s=2, color="black")
        plt.annotate(str(v['id']), (v['x'], v['y']), fontsize=2)

    # 绘制边
    for i, e in edges.iterrows():
        v1 = vertices[vertices['id'] == e['id1']].iloc[0]  # 通过边的id1和id2找到对应的顶点
        v2 = vertices[vertices['id'] == e['id2']].iloc[0]
        plt.plot([v1['x'], v2['x']], [v1['y'], v2['y']], color="black", linewidth=0.5)
        plt.annotate(str(e['id']), [(v1['x'] + v2['x']) / 2, (v1['y'] + v2['y']) / 2], fontweight="bold", fontsize=1)

    # 绘制细胞
    for i, c in cells.iterrows():
        # 假设 edges_list 中包含了边的 id 列表
        cell_edges = c['edges']
        cxs = []
        cys = []
        for edge_id in cell_edges:
            if edge_id in edges['id'].values:
                edge = edges[edges['id'] == edge_id].iloc[0]
                v1 = vertices[vertices['id'] == edge['id1']].iloc[0]
                v2 = vertices[vertices['id'] == edge['id2']].iloc[0]
                if not v1.empty and not v2.empty:
                    cxs.append(v1['x'])
                    cys.append(v1['y'])
                    cxs.append(v2['x'])
                    cys.append(v2['y'])

        else:
                print(f"Warning: edge_id {edge_id} not found in edges['id']")

        # 填充细胞的多边形区域
        plt.fill(cxs, cys, alpha=0.25, color="gray")

        # 计算质心
        cm_x = sum(cxs) / len(cxs)
        cm_y = sum(cys) / len(cys)
        plt.annotate(str(c['id']), (cm_x, cm_y))

    if len(xlim) > 0:
        plt.xlim(xlim[0], xlim[1])
    if len(ylim) > 0:
        plt.ylim(ylim[0], ylim[1])
    if mirror_y:
        plt.gca().invert_yaxis()
    if mirror_x:
        plt.gca().invert_xaxis()

    plt.axis("off")
    plt.show()
    return fig, ax


plot_mesh_v3(vertices, edges, cells, name="0", folder=".", xlim=[], ylim=[], mirror_y=False, mirror_x=False)
