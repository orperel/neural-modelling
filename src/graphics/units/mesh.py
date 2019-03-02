
class Mesh:

    def __init__(self):
        self.vertices = list()
        self.edges = set()
        self.faces = set()

        self.vertices_to_edges = {}
        self.edges_to_faces = {}

    def add_vertex(self, v):
        """
        :param v: tuple of 3 float coordinates (x,y,z)
        """
        self.vertices.append(v)

    def add_edge(self, e):
        """
        :param e: tuple of 2 vertex indices
        """
        self.edges.add(e)
        for vertex in e:
            self.vertices_to_edges.setdefault(vertex, []).append(e)

    def add_face(self, f):
        """
        :param f: tuple of n vertex indices
        """
        self.faces.add(f)

        vertices_iter = iter(f)  # Iterate over pairs of vertices
        v_first = next(vertices_iter)
        v0 = v_first
        for v1 in vertices_iter:
            edge = (v0, v1)
            self.add_edge(edge)
            self.edges_to_faces.setdefault(edge, []).append(f)
            v0 = v1

        v_last = v0
        edge = (v_last, v_first)
        self.add_edge(edge)
        self.edges_to_faces.setdefault(edge, []).append(f)

