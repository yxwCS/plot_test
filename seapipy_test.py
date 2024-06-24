import io
from dataclasses import dataclass
import numpy as np
import scipy.spatial as spatial
from copy import deepcopy

"""
The current issue is that there are still infinite regions, causing the loop not to execute. 
The problem lies in the fact that the generated grid is not a square but a cube, leading to the existence of infinite regions.
The generated regions contain -1, indicating infinite regions. How to remove these infinite regions, or in other words, how to allocate vertices, edges, and cells properly?

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

    def generate_cube_seeds(self, standard_deviation: float = 0, spatial_step: int = 1) -> list:
        grid_values = ([[(i + np.random.normal(0, standard_deviation)) * spatial_step,
                         (j + np.random.normal(0, standard_deviation)) * spatial_step,
                         (k + np.random.normal(0, standard_deviation)) * spatial_step]
                        for k in range(self.number_cells_z)
                        for j in range(self.number_cells_y)
                        for i in range(self.number_cells_x)])
        return grid_values

    def generate_voronoi_tessellation(self, seed_values: list) -> object:
        self.tessellation = spatial.Voronoi(list(seed_values))
        # print("Voronoi tessellation:", self.tessellation)  # Debug: Print Voronoi tessellation
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
# This loop is not executed 
            if len(c) != 0 and -1 not in c:  # 判断是否有无穷大区域
                # add first to close the cell_vertices
                c.append(c[0])
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
                    print("new_edge_vertices in creating lattice:", new_edge_vertices)
                    # add new edges to the global list
                    for v in range(0, len(new_edge_vertices) - 1):
                        v0 = new_edge_vertices[v]
                        v1 = new_edge_vertices[v + 1]
                        v2 = new_edge_vertices[v + 2]
                        vertex_number_1 = self.get_vertex_number(v0, new_vertices)

                        vertex_number_2 = self.get_vertex_number(v1, new_vertices)

                        # vertex_number_3 =

                        enum = self.get_enum([vertex_number_1, vertex_number_2], new_edges)

                        temp_big_edge.append(enum)
                        temp_for_cell.append(enum)
                        temp_vertex_for_cell.append(vertex_number_1)
                        temp_vertex_for_cell.append(vertex_number_2)

                    big_edge.append(temp_big_edge)

                # area_sign = self.get_cell_area_sign(temp_vertex_for_cell, new_vertices)
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
            if len(c) != 0 and -1 not in c:  # 长度不为0且不包含-1
                for ii in range(0, len(c) - 1):
                    distances.append(
                        np.linalg.norm(self.tessellation.vertices[c[ii]] - self.tessellation.vertices[c[ii + 1]]))
                distances = np.array(distances)
                if np.any(np.where(distances > max_distance, True, False)):  # 如果有任何一个大于最大距离
                    to_delete.append(c)

        for c in to_delete:
            regions.remove(c)
        return regions

    def create_example_lattice(self, voronoi_seeds_std: float = 0.15, voronoi_seeds_step: int = 20) -> tuple:
        seed_values = self.generate_cube_seeds(standard_deviation=voronoi_seeds_std, spatial_step=voronoi_seeds_step)
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
lattice = Lattice(2, 2, 2)  # 假设网格为2x2x2
vertices, edges, cells = lattice.create_example_lattice()
print("Vertices in main:", vertices)
print("Edges in main:", edges)
print("Cells in main:", cells)



# 为每个单元分配初始体积
volume_values = {k: 1.0 for k, v in cells.items()}  # 假设每个单元的体积为1.0

density_values = {k: 1.0 for k, v in edges.items()}  # 假设所有边的张力为1.0

evolver = SurfaceEvolver(vertices, edges, cells, density_values, volume_values)

fe_file = evolver.generate_fe_file()

# 添加gogo和inner脚本
# fe_file.write('''
# gogo := { g 5; r; g 12; r; g 12; hessian_normal; hessian; hessian; }
# inner := { show facets ff where sum(ff.body,1) == 2 }
# ''')

# 保存文件
input_file_path = r"C:\Users\takeoff\Desktop\GRINDING\3D_test.fe"

evolver.save_fe_file(input_file_path)
# print(fe_file.getvalue())
