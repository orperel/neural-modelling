from abc import ABC
from panda3d.core import LVector3
from panda3d.core import GeomVertexFormat, GeomVertexData
from panda3d.core import Geom, GeomTriangles, GeomLines, GeomPoints, GeomVertexWriter, GeomNode


def _static_id_allocator():
    next_id = 0
    while True:
        yield 'renderable_' + str(next_id)
        next_id += 1


class AbstractRenderable(ABC):

    id_allocator = _static_id_allocator()  # ID generator, joint for all Renderable objects

    def __init__(self, default_color=None, alpha=1.0):

        node_name = next(AbstractRenderable.id_allocator)
        format = GeomVertexFormat.getV3n3cp()
        self.vertex_data = GeomVertexData(node_name, format, Geom.UHDynamic)

        self.vertex_writer = GeomVertexWriter(self.vertex_data, 'vertex')
        self.normal_writer = GeomVertexWriter(self.vertex_data, 'normal')
        self.color_writer = GeomVertexWriter(self.vertex_data, 'color')

        self.points_list = GeomPoints(Geom.UHDynamic)
        self.edges_list = GeomLines(Geom.UHDynamic)
        self.triangle_list = GeomTriangles(Geom.UHDynamic)

        self.mesh_node = GeomNode(node_name)
        self.alpha = alpha

        if default_color is None:
            default_color = (1.0, 0.0, 0.0, alpha)   # Red
        self.default_color = default_color

    @staticmethod
    def _normalized(*args):
        """ Helper function for normalizing inline """
        myVec = LVector3(*args)
        myVec.normalize()
        return myVec

    def _add_vertex_data(self, x, y, z, color=None):
        self.vertex_writer.addData3(x, y, z)
        self.normal_writer.addData3(self._normalized(2 * x - 1, 2 * y - 1, 2 * z - 1))
        vertex_color = color or self.default_color
        vertex_color = (vertex_color[0], vertex_color[1], vertex_color[2], self.alpha)
        self.color_writer.addData4f(*vertex_color)

    def _add_point(self, v1):
        """
        Adds a new point to the mesh (single vertex rendered as a point)
        :param v1: Index of vertex #1
        """
        self.points_list.addVertex(v1)

    def _add_edge(self, v1, v2):
        """
        Adds a new edge to the mesh
        :param v1: Index of vertex #1
        :param v2: Index of vertex #2
        """
        self.edges_list.addVertices(v1, v2)

    def _add_triangle(self, v1, v2, v3):
        """
        Adds a new triangle to the mesh
        :param v1: Index of vertex #1
        :param v2: Index of vertex #2
        :param v3: Index of vertex #3
        """
        self.triangle_list.addVertices(v1, v2, v3)

    def render(self):
        self.mesh_node.remove_all_geoms()

        mesh_points = Geom(self.vertex_data)
        mesh_edges = Geom(self.vertex_data)
        mesh_faces = Geom(self.vertex_data)

        mesh_points.addPrimitive(self.points_list)
        mesh_edges.addPrimitive(self.edges_list)
        mesh_faces.addPrimitive(self.triangle_list)

        self.mesh_node.addGeom(mesh_points)
        self.mesh_node.addGeom(mesh_edges)
        self.mesh_node.addGeom(mesh_faces)
