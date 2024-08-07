import io
from dataclasses import dataclass
import numpy as np
import scipy.spatial as spatial
from copy import deepcopy
# import seapipy as sep
import pytest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import pyvoro
import pytest
# 30 June. Fe file is generated. The plot in Surface Evolver is wroking but something is wrong
# Guess is the distance calculation in remove_infinite_regions function needs fixing
# Does the way generating coordinates need to be fixed?




"""
lattice = sep.lattice_class.Lattice(10, 10)

vertices, edges, cells = lattice.create_example_lattice()

volume_values = {k: 500 for k, v in cells.items()}

initial_edges_tensions = lattice.get_normally_distributed_densities(edges)

# 创建 Surface Evolver 对象
se_object = sep.surface_evolver.SurfaceEvolver(vertices, edges, cells, initial_edges_tensions, volume_values, polygonal=False)
se_file = se_object.generate_fe_file()
# print(se_object.fe_file.getvalue())

# 定义输出目录和文件名
output_dir = r"C:\\Users\\takeoff\\Desktop\\GRINDING"
file_name = "test_output"
"""

@dataclass
class Lattice:
    number_cells_x: int
    number_cells_y: int
    number_cells_z: int

    tessellation: object = None

    def __post_init__(self):
        ...

    def generate_square_seeds(self, standard_deviation: float = 0, spatial_step: int = 1) -> list:
        grid_values = ([[(i + np.random.normal(0, standard_deviation)) * spatial_step,
                         (j + np.random.normal(0, standard_deviation)) * spatial_step]
                        for j in range(self.number_cells_y)
                        for i in range(self.number_cells_x)])
        return grid_values

    def generate_cube_seeds(self, standard_deviation: float = 0, spatial_step: int = 1) -> list:
        # 3D中是不是，不能生成网格？
            
        grid_values = ([[(i + np.random.normal(0, standard_deviation)) * spatial_step,
                         (j + np.random.normal(0, standard_deviation)) * spatial_step,
                         (k + np.random.normal(0, standard_deviation)) * spatial_step]
                        for k in range(self.number_cells_z)
                        for j in range(self.number_cells_y)
                        for i in range(self.number_cells_x)])
        # grid ??
        unique_grid_values = [list(x) for x in set(tuple(x) for x in grid_values)]

        min_seed = np.min(unique_grid_values, axis=0)
        max_seed = np.max(unique_grid_values, axis=0)
        limits = [[min_seed[0] - 10, max_seed[0] + 10],
                  [min_seed[1] - 10, max_seed[1] + 10],
                  [min_seed[2] - 10, max_seed[2] + 10]]

        print("Voronoi limits:", limits)  # Debug: Print Voronoi limits

        # 计算 Voronoi 图
        voronoi = pyvoro.compute_voronoi(unique_grid_values, limits, 2)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        rng = np.random.default_rng(11)

        for vnoicell in voronoi:
            faces = []
            vertices = np.array(vnoicell['vertices'])
            for face in vnoicell['faces']:
                faces.append(vertices[np.array(face['vertices'])])

            polygon = Poly3DCollection(faces, alpha=0.5,
                                       facecolors=rng.uniform(0, 1, 3),
                                       linewidths=0.5, edgecolors='black')
            ax.add_collection3d(polygon)

        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_zlim(limits[2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
        return unique_grid_values

    def plot_seeds(self, seeds):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        seeds = np.array(seeds)
        ax.scatter(seeds[:, 0], seeds[:, 1], seeds[:, 2], c='r', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def generate_voronoi_tessellation(self, seed_values: list) -> object:
        self.tessellation = spatial.Voronoi(list(seed_values))
        # print("Voronoi tessellation:", self.tessellation)  # Debug: Print Voronoi tessellation
        return self.tessellation

    # 这里定义voronoi的绘制函数
    def plot_voronoi_tessellation(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 绘制种子点
        seed_values = np.array(self.tessellation.points)
        ax.scatter(seed_values[:, 0], seed_values[:, 1], seed_values[:, 2], c='r')
        # 绘制Voronoi顶点
        ax.scatter(self.tessellation.vertices[:, 0], self.tessellation.vertices[:, 1], self.tessellation.vertices[:, 2], c='b')
        plt.show()

    def plot_tessellation(self):
        # self.tessellation = Voronoi(seeds)
        rng = np.random.default_rng()
        polygons = []
        for ri, region in enumerate(self.tessellation.regions):
            # only plot those polygons for which all vertices are defined
            if np.all(np.asarray(region) >= 0) and len(region) > 0:
                poly = []
                for rv in self.tessellation.ridge_vertices:
                    if np.isin(rv, region).all():
                        poly.append(self.tessellation.vertices[rv])
                polygons.append(poly)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for poly in polygons:
            polygon = Poly3DCollection(poly, alpha=0.5,
                                       facecolors=rng.uniform(0, 1, 3),
                                       linewidths=0.5, edgecolors='black')
            ax.add_collection3d(polygon)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        plt.savefig("Voronoi_3D_3.jpg", bbox_inches='tight', dpi=100)
        plt.show()

    def plot_voronoi_3d(self, seeds):
        voronoi = pyvoro.compute_voronoi(seeds, [[0, 1], [0, 1], [0, 1]], 1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        rng = np.random.default_rng(11)

        for vnoicell in voronoi:
            faces = []
            vertices = np.array(vnoicell['vertices'])
            for face in vnoicell['faces']:
                faces.append(vertices[np.array(face['vertices'])])

            polygon = Poly3DCollection(faces, alpha=0.5,
                                       facecolors=rng.uniform(0, 1, 3),
                                       linewidths=0.5, edgecolors='black')
            ax.add_collection3d(polygon)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def returun_voronoi_tessellation(self):
        return self.tessellation

    def create_lattice_elements(self) -> tuple:
        """
        Create the vertices, edges and cells from the scipy.spatial.Voronoi() objects

        :return: Vertices, edges and cell's list
        :rtype: tuple
        """
        new_vertices = {}
        new_cells = {}
        new_edges = {}
        regions = deepcopy(self.tessellation.regions)
        big_edge = []
        print("regions before removing in lattice creating", regions)
        cnum = 1
        regions = self.remove_infinite_regions(regions)
        print("regions after in lattice creating", regions)
        for c in regions:
            temp_for_cell = []
            temp_vertex_for_cell = []
            if not len(c) != 0 and -1 not in c:
                print("存在无穷大区域")
# 已解决   现在的问题是还存在无穷大区域，导致不会进循环
# 问题出现在生成的网格不是正方形，而是立方体，导致无穷大区域的存在
# 生成的regions中包含-1，表示无穷大区域，怎么去掉这个无穷大区域或者说怎么分配vertices，edges，cells
            if len(c) != 0 and -1 not in c:  # 判断是否有无穷大区域，这个循环导致进不去
                # add first to close the cell_vertices
                if c[0] != c[-1]:
                    c.append(c[0])  # 为了闭合顶点
                for ii in range(0, len(c) - 1):
                    temp_big_edge = []

                    x_coordinate = np.around(np.linspace(round(self.tessellation.vertices[c[ii]][0], 3),
                                                         round(self.tessellation.vertices[c[ii + 1]][0], 3), 2), 3)

                    y_coordinate = np.around(self.line_eq(self.tessellation.vertices[c[ii]],
                                                          self.tessellation.vertices[c[ii + 1]],
                                                          x_coordinate), 3)

                    z_coordinate = np.around(np.linspace(round(self.tessellation.vertices[c[ii]][2], 3),
                                                         round(self.tessellation.vertices[c[ii + 1]][2], 3), 2), 3)

                    new_edge_vertices = list(zip(x_coordinate, y_coordinate, z_coordinate))

                    # v0 = self.tessellation.vertices[c[ii]]
                    # v1 = self.tessellation.vertices[c[ii + 1]]
                    # t_values = np.linspace(0, 1, num=2)  # 生成从 0 到 1 的均匀间隔点
                    # new_edge_vertices = self.line_eq_v1(v0, v1, t_values)
                    # print("new_edge_vertices in creating lattice:", new_edge_vertices)

                    # pass
                    # add new edges to the global list
                    for v in range(0, len(new_edge_vertices) - 1):
                        v0 = new_edge_vertices[v]
                        v1 = new_edge_vertices[v + 1]
                        # v0 = self.tessellation.vertices[c[ii]]
                        # v1 = self.tessellation.vertices[c[ii + 1]]
                        # new_edge_vertices = [v0, v1]
                        # v2 = new_edge_vertices[v + 2]
                        vertex_number_1 = self.get_vertex_number(v0, new_vertices)  # 通过坐标获取顶点编号
                        vertex_number_2 = self.get_vertex_number(v1, new_vertices)


                        # vertex_number_1 = self.get_vertex_number_v1(v0, new_vertices)  # 通过坐标获取顶点编号
                        #
                        # vertex_number_2 = self.get_vertex_number_v1(v1, new_vertices)

                        # vertex_number_3 =

                        enum = self.get_enum([vertex_number_1, vertex_number_2], new_edges)

                        temp_big_edge.append(enum)
                        temp_for_cell.append(enum)
                        temp_vertex_for_cell.append(vertex_number_1)
                        temp_vertex_for_cell.append(vertex_number_2)

                    big_edge.append(temp_big_edge)

                area_sign = self.get_cell_area_sign(temp_vertex_for_cell, new_vertices)  # 通过高斯公式计算多边形的方向
                new_cells[-1 * cnum * area_sign] = temp_for_cell
                # new_cells[cnum] = temp_for_cell
                # calculate cell_vertices centroid
                cnum += 1
        print("new_edges in creating lattice:", new_edges)
        print("new_vertices in creating lattice:", new_vertices)
        print("new_cells in creating lattice:", new_cells)

        return new_vertices, new_edges, new_cells

    def remove_infinite_regions(self, regions: list, max_distance: float = 50) -> list:
        to_delete = []
        for c in regions:
            distances = []
            if len(c) != 0 and -1 not in c:  # 长度不为0且不包含-1, 目前没有无穷大区域
                print("c in remove infinite regions:", c)  # 运行结果不显示这一行，说明没有无穷大区域
                for ii in range(0, len(c) - 1):
                    distances.append(
                        np.linalg.norm(self.tessellation.vertices[c[ii]] - self.tessellation.vertices[c[ii + 1]]))
                distances = np.array(distances)
                if np.any(np.where(distances > max_distance, True, False)):  # 如果有任何一个大于最大距离
                    to_delete.append(c)

        for c in to_delete:
            regions.remove(c)
        print("regions after removing in remove infinite regions:", regions)
        return regions

    def get_cell_area_sign(self, cell: list, all_vertices: dict) -> int:
        """
        Get the orientation of the polygon for a cell_vertices using the sign of the area through gauss formula

        :param cell: Vertices of the cell_vertices to determine orientation
        :type cell: list
        :param all_vertices: Dictionary with all the vertices in the lattice
        :type all_vertices: dict
        :return: Sign of the cell_vertices area to determine orientation of the polygon
        :rtype: int
        """
        return int(np.sign(self.get_cell_area(cell, all_vertices)))  #

    def create_example_lattice(self, voronoi_seeds_std: float = 0.15, voronoi_seeds_step: int = 20) -> tuple:
        seed_values = self.generate_cube_seeds(standard_deviation=voronoi_seeds_std, spatial_step=voronoi_seeds_step)
        # the voronoi generated from this seed contains -1 in every region

        print("Generated seed values:", seed_values)  # Debug: Print generated seed values
        self.generate_voronoi_tessellation(seed_values)
        print("Voronoi vertices:", self.tessellation.vertices)  # Debug: Print Voronoi vertices
        print("Voronoi regions(self.tessellation regions):", self.tessellation.regions)  # Debug: Print Voronoi regions

        vertices, edges, cells = self.create_lattice_elements()

        print("Vertices in create example lattice:", vertices)  # Debug: Print created vertices
        print("Edges in create example lattice:", edges)
        return vertices, edges, cells

    @staticmethod
    def get_vertex_number(vertex: list, vertices: dict) -> int:
        if vertex in vertices.values():
            vertex_number = list(vertices.keys())[list(vertices.values()).index(vertex)]
        else:
            if len(vertices) > 0:
                vertex_number = max(vertices.keys()) + 1
            else:
                vertex_number = 1
            vertices[vertex_number] = vertex
        return vertex_number

    @staticmethod
    def get_vertex_number_v1(vertex: np.ndarray, vertices: dict) -> int:
        for key, value in vertices.items():
            if np.array_equal(vertex, value):
                return key
        if len(vertices) > 0:
            vertex_number = max(vertices.keys()) + 1
        else:
            vertex_number = 1
        vertices[vertex_number] = vertex
        return vertex_number


    @staticmethod
    def get_enum(edge: list, edges: dict) -> int:
        if edge in edges.values():
            enum = list(edges.keys())[list(edges.values()).index(edge)]
        elif edge[::-1] in edges.values():
            enum = - list(edges.keys())[list(edges.values()).index(edge[::-1])]
        else:
            if len(edges) > 0:
                enum = max(edges.keys()) + 1
            else:
                enum = 1
            edges[enum] = [edge[0], edge[1]]
        return enum

    @staticmethod
    def line_eq(p0: float, p1: float, x: np.ndarray) -> list:
        """
        Get the value of linear function that goes through p0 and p1 at x

        :param p0: First point to join
        :type p0: float
        :param p1: Second point to join
        :type p0: float
        :param x: Independent variable of the equation
        :type p0: list
        :return: Value of y(x) for a line that goes through p0 and p1
        """
        p0 = np.around(p0, 3)
        p1 = np.around(p1, 3)
        m = (p1[1] - p0[1]) / (p1[0] - p0[0])
        return p0[1] + m * (x - p0[0])

    @staticmethod
    def line_eq_v1(p0, p1, x) -> list:
        p0 = np.around(p0, 3)
        p1 = np.around(p1, 3)

        # Compute the direction vector from p0 to p1
        direction = p1 - p0
        if np.isclose(direction[0], 0):
            direction[0] = 1e-9
        # Compute the points on the line at each t
        return p0 + x[:, np.newaxis] * direction

    @staticmethod
    def get_cell_area(cell_vertices: list, vertices: dict) -> float:
        """
        Area of a polygon using the shoelace formula

        :param cell_vertices: List of vertex's id of the polygon
        :param vertices: Dictionary of vertices in the system
        :return: Area of the polygon. Positive number indicates clockwise orientation, negative number counterclockwise.
        """
        x = [vertices[i][0] for i in cell_vertices]
        y = [vertices[i][1] for i in cell_vertices]
        # z =
        return 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


@dataclass
class SurfaceEvolver:
    """
    Class to interact with the Surface Evolver file buffer.

    :param vertices: Vertices in the tessellation
    :type vertices: dict
    :param edges: Edges in the tessellation
    :type edges: dict
    :param cells: Cells in the tessellation
    :type cells: dict
    :param density_values: Initial line tensions for the edges
    :type density_values: dict
    :param volume_values: Initial target volumes in the tessellation
    :type volume_values: dict
    :param polygonal: Whether to use polygons or allowed curved edges
    :type polygonal: bool, optional
    """
    vertices: dict
    edges: dict
    cells: dict

    density_values: dict
    volume_values: dict

    polygonal: bool = True

    def __post_init__(self):
        self.fe_file = io.StringIO()
        self.density_values = {key: round(value, 3) for key, value in self.density_values.items()}

    def generate_fe_file(self) -> io.StringIO:
        """
        Generate the initial Surface Evolver slate to write into

        :return: Initialized Surface Evolver slate
        :rtype: io.StringIO()
        """
        print(self.vertices.items())
        self.fe_file.write("SPACE_DIMENSION 3 \n")  # 修改为3D
        self.fe_file.write("SCALE 0.005 FIXED\n")
        self.fe_file.write("STRING \n")
        self.fe_file.write("\n")
        self.fe_file.write("vertices \n")

        for k, v in self.vertices.items():
            self.fe_file.write(f"{k}   {round(v[0], 3)} {round(v[1], 3)} {round(v[2], 3)}\n")  # 添加z坐标
        self.fe_file.write("\n")
        self.fe_file.write("edges \n")
        for k, v in self.edges.items():
            lambda_val = self.density_values[k]
            self.fe_file.write(f"{abs(k)}   {v[0]}   {v[1]}   density {lambda_val}\n")

        self.fe_file.write("\n")
        self.fe_file.write("faces \n")
        for k, v in self.cells.items():
            str_value = " ".join(str(vv) for vv in v)
            self.fe_file.write(f"{abs(k)}   {str_value} \n")

        self.fe_file.write("\n")
        self.fe_file.write("bodies \n")
        for k, v in self.cells.items():
            self.fe_file.write(f"{abs(k)}   {k}    VOLUME {self.volume_values[k]} \n")

        self.fe_file.write("\n \n")
        self.fe_file.write("read \n \n")
        self.fe_file.write("show_all_edges off \n")
        self.fe_file.write("metric_conversion off \n")
        self.fe_file.write("autorecalc on \n")
        self.fe_file.write("gv_binary off \n")
        self.fe_file.write("gravity off \n")
        self.fe_file.write("ii := 0; \n")

        if not self.polygonal:
            self.add_refining_triangulation(3)
        return self.fe_file

    def add_vertex_averaging(self, how_many: int = 1) -> io.StringIO:
        """
        Add vertex averaging using the V Surface Evolver function, at the end of the Surface Evolver slate

        :param how_many: Number of times the averaging should be done
        :return: Current Surface Evolver slate
        :rtype: io.StringIO()
        """
        self.fe_file.write(f"V {how_many}; \n")
        return self.fe_file

    def add_refining_triangulation(self, how_many: int = 1) -> io.StringIO:
        """
        Add a mesh refinement using the r Surface Evolver function, at the end of the Surface Evolver slate

        :param how_many: Number of times the refinement should be done
        :type how_many: int
        :return: Current Surface Evolver slate
        :rtype: io.StringIO()
        """
        self.fe_file.write(f"r {how_many}; \n")
        return self.fe_file

    def change_scale(self, new_scale: float) -> io.StringIO:
        """
        Change the scale of the Surface Evolver simulation at the end of the Surface Evolver slate

        :param new_scale: Numerical value of the new scale
        :type new_scale: float
        :return: Current Surface Evolver slate
        :rtype: io.StringIO()
        """
        self.fe_file.write(f"scale := {new_scale}; \n")
        return self.fe_file

    def evolve_system(self, steps: int = 1) -> io.StringIO:
        """
        Add evolution function using the Surface Evolver go function

        :param steps: Number of steps to evolve for
        :type steps: int
        :return: Current Surface Evolver slate
        :rtype: io.StringIO()
        """
        self.fe_file.write(f"g {steps}; \n")
        return self.fe_file

    def add_t1_swaps(self, max_size: float = 0.1) -> io.StringIO:
        """
        Add check for T1 swaps using the Surface Evolver t1_edgeswap function

        :param max_size: Maximum size of interfaces before making a T1 swap
        :type max_size: float
        :return: Current Surface Evolver slate
        :rtype: io.StringIO()
        """
        self.fe_file.write(f"t1_edgeswap edge where length < {max_size}; \n")
        return self.fe_file

    def initial_relaxing(self, evolve_step: int = 2500, averaging: int = 100) -> io.StringIO:
        """
        Initial standard relaxing with vertex averaging and scale change followed by evolution

        :param evolve_step: Number of steps to evolve at each scale change
        :type evolve_step: int
        :param averaging: Number of vertex averagings to perform
        :type evolve_step: int
        :return: Current Surface Evolver slate
        :rtype: io.StringIO
        """
        self.add_vertex_averaging(averaging)
        self.change_scale(0.25)
        self.evolve_system(evolve_step)
        self.add_vertex_averaging(averaging)
        self.change_scale(0.1)
        self.evolve_system(evolve_step)
        self.add_vertex_averaging(averaging)
        self.change_scale(0.01)
        return self.fe_file

    def evolve_relaxing(self, number_of_times: int = 1, steps: int = 1, max_size: float = 0.1) -> io.StringIO:
        """
        Evolve the system a fixed number of steps and perform T1 swaps after a definite number of times

        :param number_of_times: Number of times the evolution plus T1 swaps happen
        :type number_of_times: int
        :param steps: Number of evolution step for all iterations
        :type steps: int
        :param max_size: Maximum size allowed for membranes before a T1 happens
        :type max_size: float
        :return: Current Surface Evolver slate
        :rtype: io.StringIO()
        """
        for _ in range(0, number_of_times):
            self.evolve_system(steps)
            self.add_t1_swaps(max_size)
        return self.fe_file

    def save_one_step(self, output_directory: str, file_name: str) -> io.StringIO:
        """
        Add a savepoint to the Surface Evolver slate using the sprintf function

        :param output_directory: Folder to save the file into
        :type output_directory: str
        :param file_name: Name of the file to be saved
        :type file_name: str
        :return: Current Surface Evolver slate
        :rtype: io.StringIO()
        """
        self.fe_file.write(f'ff := sprintf "{output_directory}\\{file_name}%d.dmp",ii; dump ff; ii+=1; \n')
        return self.fe_file

    def save_many_steps(self, output_directory, file_name, max_steps, time_step=1, averaging=1, max_size=0.1):
        self.fe_file.write('while ii < ' + str(max_steps) + ' do { '
                                                            'g ' + str(time_step) + '; V ' + str(averaging) +
                           '; t1_edgeswap edge where length < ' + str(max_size) + '; '
                           'ff := sprintf "' + output_directory + "/" + file_name + '%d.dmp",ii;'
                                                                                    ' dump ff; ii:=ii+1} \n')
        return self.fe_file

    def save_fe_file(self, file_name: str) -> bool:
        """
        Save the Surface Evolver slate to disk

        :param file_name: Name of the file to be saved
        :type file_name: str
        :return: Success state of the saving
        :rtype: bool
        """
        self.fe_file.write('q; \n')
        with open(f'{file_name}', mode='w') as f:
            print(self.fe_file.getvalue(), file=f)
        return True

    def change_line_tensions(self, new_tensions: dict) -> io.StringIO:
        """
        Set new densities for the system's  membranes. Setting is done using the dictionary key as membrane id and the
        value as the new tension

        :param new_tensions: Dictionary with new tension values
        :type new_tensions: dict
        :return: Current Surface Evolver slate
        :rtype: io.StringIO
        """
        for eid, tension in new_tensions.items():
            self.fe_file.write(f"set edges density {tension} where original == {abs(eid)}; \n")
        return self.fe_file

# create the lattice


# 生成3D晶格

# 怎么样只选择两个参数传入
lattice = Lattice(3, 3, 3)  # 假设网格为2x2x2
# 这里提取voronoi图，并绘制


vertices, edges, cells = lattice.create_example_lattice()
print("Vertices in main:", vertices)
print("Edges in main:", edges)
print("Cells in main:", cells)
# lattice.plot_seeds()
# lattice.plot_tessellation()

# 为每个单元分配初始体积
volume_values = {k: 1.0 for k, v in cells.items()}  # 假设每个单元的体积为1.0

density_values = {k: 1.0 for k, v in edges.items()}  # 假设所有边的张力为1.0


# 生成Surface Evolver对象
# evolver = sep.surface_evolver.SurfaceEvolver(vertices, edges, cells, density_values, volume_values, polygonal=False)




# 添加gogo和inner脚本
# fe_file.write('''
# gogo := { g 5; r; g 12; r; g 12; hessian_normal; hessian; hessian; }
# inner := { show facets ff where sum(ff.body,1) == 2 }
# ''')


# 得到所有的信息，现在生成文件
# print(fe_file.getvalue())


# # # 保存文件
evolver = SurfaceEvolver(vertices, edges, cells, density_values, volume_values)

# 写入文件，初始松弛，然后保存文件
evolver.generate_fe_file()
evolver.initial_relaxing()
input_file_path = r"C:\Users\takeoff\Desktop\GRINDING\3D_testv5.fe"
evolver.save_fe_file(input_file_path)
# print(fe_file.getvalue())


# 打印voronoi图的基本信息
voronoi = lattice.returun_voronoi_tessellation()


def plot_voronoi_3d(voronoi):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制Voronoi种子点
    ax.scatter(voronoi.points[:, 0], voronoi.points[:, 1], voronoi.points[:, 2], c='r', marker='o', label='Seeds')

    # 绘制Voronoi顶点
    ax.scatter(voronoi.vertices[:, 0], voronoi.vertices[:, 1], voronoi.vertices[:, 2], c='b', marker='^',
               label='Vertices')

    # 绘制Voronoi边
    for simplex in voronoi.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            ax.plot(voronoi.vertices[simplex, 0], voronoi.vertices[simplex, 1], voronoi.vertices[simplex, 2], 'k-')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.legend()
    plt.show()


#
# lattice = Lattice(2, 2, 2)
# vertices, edges, cells = lattice.create_example_lattice()
# voronoi = lattice.generate_voronoi_tessellation(seed_values)
#
# # plot_voronoi_3d(voronoi)
# plot_voronoi_3d(voronoi)

# 使用pytest测试line_eq函数
def test_line_eq():
    p0 = [0, 0, 0]
    p1 = [1, 1, 1]
    x = np.array([0.5, 0.75])
    y = lattice.line_eq(p0, p1, x)
    assert np.allclose(y, [0.5, 0.75])

    p0 = [0, 0, 0]
    p1 = [0, 1, 0]
    x = np.array([0.5, 0.75])
    y = lattice.line_eq(p0, p1, x)
    assert np.allclose(y, [0.5, 0.75])

    p0 = [0, 0, 0]
    p1 = [0, 0, 1]
    x = np.array([0.5, 0.75])
    y = lattice.line_eq(p0, p1, x)
    assert np.allclose(y, [0.5, 0.75])

    p0 = [0, 0, 0]
    p1 = [1, 1, 1]
    x = np.array([0.5, 0.75])
    y = lattice.line_eq(p0, p1, x)
    assert np.allclose(y, [0.5, 0.75])

    p0 = [0, 0, 0]
    p1 = [1, 1, 1]
    x = np.array([0.5, 0.75])
    y = lattice.line_eq(p0, p1, x)
    assert np.allclose(y, [0.5, 0.75])