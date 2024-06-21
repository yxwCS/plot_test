import forsys as fsys
import os
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
import re
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objs as go
from forsys.surface_evolver import SurfaceEvolver


filepath = r"C:\Users\takeoff\Desktop\GRINDING\test_output0.dmp"  # read the file

# read the Surface Evolver file(.dmp here)
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

# read vertices
# and type fields will have been populated.
class Vertex:
    id: int
    x: float
    y: float

    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y
        self.ownEdges = []
        self.ownCells = []
        self.own_big_edges = []


    def get_coords(self) -> list:

        return [self.x, self.y]

    def add_cell(self, cid: int) -> bool:
        if cid in self.ownCells:
            return False
        else:
            self.ownCells.append(cid)
            return True

    def remove_cell(self, cid: int) -> list:
        self.ownCells.remove(cid)
        return self.ownCells

    def add_edge(self, eid: int) -> bool:
        if eid in self.ownEdges:
            print(eid, self.ownEdges)
            print("edge already in vertex")
            return False
        else:
            self.ownEdges.append(eid)
            return True

    def remove_edge(self, eid: int) -> list:
        self.ownEdges.remove(eid)
        return self.ownEdges


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
vertices_dict = {}
for _, r in vertices.iterrows():  # r是每一行vertices的数据
    vertices_dict[int(r.id)] = Vertex(int(r.id), float(r.x), float(r.y))


# read edges
class Edge:
    def __init__(self, id: int, v1: Vertex, v2: Vertex):
        self.id = id
        self.v1 = v1
        self.v2 = v2
        self.tension = 0
        self.gt = 0
        self.verticesArray = [self.v1, self.v2]
        for v in self.verticesArray:
            v.add_edge(self.id)
        assert self.v1.id != self.v2.id, f"Edge {self.id} with the same vertex twice"

    def __post_init__(self):
        self.verticesArray = [self.v1, self.v2]
        for v in self.verticesArray:
            v.add_edge(self.id)
        assert self.v1.id != self.v2.id, f"edge {self.id} with the same vertex twice"

    def __del__(self):
        for v in self.verticesArray:
            if self.id in v.ownEdges:
                v.remove_edge(self.id)





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

edges_dict = {}
for _, r in edges.iterrows():
    edges_dict[int(r.id)] = Edge(int(r.id), vertices_dict[int(r.id1)], vertices_dict[int(r.id2)])
    edges_dict[int(r.id)].gt = round(edges.loc[edges['id'] == int(r.id)]['force'].iloc[0], 4)
# return edges


# read faces(cells)
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
        splitted = lines[i].split()  # split函数默认以空格分割，返回一个列表
        if len(splitted) > 0:
            pressures[int(splitted[0])] = float(splitted[7])  # 读取每一行的数据的第一个和第八个数据
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
# print(cells['edges'])
# cells["vertices"] = edges_list
cells["pressures"] = pressure_dict.values()
cells_dict = {}


def calculate_centroid(vertices):
    if not vertices:
        raise ValueError("Vertices list is empty")

    x_coords = [vertex.x for vertex in vertices]
    y_coords = [vertex.y for vertex in vertices]

    centroid_x = sum(x_coords) / len(vertices)
    centroid_y = sum(y_coords) / len(vertices)

    return centroid_x, centroid_y


class Cell:
    def __init__(self, id: int, vertices: list, gt_pressure: float):
        self.id = id
        self.vertices = vertices
        self.gt_pressure = gt_pressure
        self.ownEdges = []
        self.own_big_edges = []
        self.is_border = False
        # define center_method as dlite
        self.center_method = "dlite"

        for v in self.vertices:
            v.add_cell(self.id)
        # self.center_x, self.center_y = calculate_centroid(vertices=self.vertices)

    # def calculate_perimeter(self):
    #     self.perimeter = 0
    #     for i in range(len(self.vertices)):
    #         diffx = self.vertices[i].x - self.vertices[(i + 1) % len(self.vertices)].x
    #         diffy = self.vertices[i].y - self.vertices[(i + 1) % len(self.vertices)].y
    #         self.perimeter += np.sqrt(diffx ** 2 + diffy ** 2)
    #
    # def get_edges(self):
    #     cell_edges = []
    #     for vnum in range(0, len(self.vertices) - 1):
    #         local_edges = list(set(self.vertices[vnum].ownEdges) &
    #                            set(self.vertices[vnum + 1].ownEdges))
    #         cell_edges.append(local_edges[0])
    #     return cell_edges
    #
    # def get_vertices(self):
    #     return self.vertices
    #
    # def get_perimeter(self):
    #     return self.perimeter
    #
    # def is_border(self):
    #     return self.is_border
    #
    # def add_edge(self, eid: int) -> bool:
    #     if eid in self.ownEdges:
    #         return False
    #     else:
    #         self.ownEdges.append(eid)
    #         return True
    #
    # def remove_edge(self, eid: int) -> list:
    #     self.ownEdges.remove(eid)
    #     return self.ownEdges
    #
    # def add_big_edge(self, beid: int) -> bool:
    #     if beid in self.own_big_edges:
    #         return False
    #     else:
    #         self.own_big_edges.append(beid)
    #         return True
    #
    # def remove_big_edge(self, beid: int) -> list:
    #     self.own_big_edges.remove(beid)
    #     return self.own_big_edges
    #
    # def is_border(self):
    #     return self.is_border
    #
    # def calculate_neighbors(self) -> list:
    #     current_cells = set()
    #     for vertex in self.vertices:
    #         [current_cells.add(cell_id) for cell_id in vertex.ownCells]
    #     current_cells = list(current_cells)
    #     current_cells.remove(self.id)

for _, r in cells.iterrows():
    vlist = [edges_dict[e].v1 if e > 0 else edges_dict[-e].v2 for e in r.edges]
    gt_pressure = round(r["pressures"], 4)
    cells_dict[int(r.id)] = Cell(int(r.id), vlist, gt_pressure=gt_pressure)


    def get_cm(self) -> list:
        """
        Get the centroid of the cell

        :return:[x, y] list with the centroid position
        :rtype: list
        """
        return [np.mean([v.x for v in self.vertices]), np.mean([v.y for v in self.vertices])]


def plot_mesh(vertices: dict, edges: dict, cells: dict,
              name: str="0", folder: str=".",
              xlim: list=[], ylim: list=[],
              mirror_y: bool = False,
              mirror_x: bool = False) -> None:
    """Generate a plot of the mesh. Useful for visualizing IDs of cells, edges and vertices in the system.

    :param vertices: Dictionary of vertices
    :type vertices: dict
    :param edges: Dictionary of edges
    :type edges: dict
    :param cells: Dictionary of cells
    :type cells: dict
    :param name: Name of output plot, defaults to "0"
    :type name: str, optional
    :param folder: Path to folder to save the plot, defaults to "."
    :type folder: str, optional
    :param xlim: Array of [x_min, x_max] to "zoom" in, defaults to []
    :type xlim: list, optional
    :param ylim: Array of [y_min, y_max] to "zoom" in, defaults to []
    :type ylim: list, optional
    :param mirror_y: If True the tissue is plotted as a mirror image in the Y axis, defaults to False
    :type mirror_y: bool, optional
    """
    fig, ax = plt.subplots(1,1)
    if not os.path.exists(folder):
        os.makedirs(folder)
    to_save = os.path.join(folder, name)
    for v in vertices.values():
        plt.scatter(v.x, v.y, s=2, color="black")
        plt.annotate(str(v.id), [v.x, v.y], fontsize=2)

    for e in edges.values():
        plt.plot([e.v1.x, e.v2.x], [e.v1.y, e.v2.y], color="black", linewidth=0.5)
        # edges.values.v1.x
        plt.annotate(str(e.id), [(e.v1.x +  e.v2.x)/2 , (e.v1.y + e.v2.y)/2],
                     fontweight="bold", fontsize=1)  # 计算的是边的中点，将注释放在中点
        # 注释为边的id

    for c in cells.values():  # 这个循环计算了每个细胞的质心，质心 = 所有顶点的坐标之和/顶点数
        # 计算质心的坐标,不调用get_cm方法
        cm = [np.mean([v.x for v in c.vertices]), np.mean([v.y for v in c.vertices])]


        cxs = [v.x for v in c.vertices]  # 获取细胞顶点的x坐标，cells中包含了顶点信息
        # cells.values.vertices.x
        cys = [v.y for v in c.vertices]
        # 获取细胞顶点的位置，cells中包含了顶点信息
        plt.fill(cxs, cys, alpha=0.25, color="gray")
        plt.annotate(str(c.id), [cm[0], cm[1]])  # 将细胞的id放在质心处

    if len(xlim) > 0:
        plt.xlim(xlim[0], xlim[1])
    if len(ylim) > 0:
        plt.ylim(ylim[0], ylim[1])
    if mirror_y:
        plt.gca().invert_yaxis()
    if mirror_x:
        plt.gca().invert_xaxis()

    plt.axis("off")
    return fig, ax

plot_mesh(vertices_dict, edges_dict, cells_dict)
plt.show()
