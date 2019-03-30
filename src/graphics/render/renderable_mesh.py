from graphics.render.abstract_renderable import AbstractRenderable


class RenderableMesh(AbstractRenderable):

    def __init__(self, mesh=None, default_color=None, alpha=1.0):
        super().__init__(default_color, alpha)

        if mesh is not None:
            self.add_mesh(mesh)

    def add_mesh(self, mesh):
        for vertex in mesh.vertices:
            x, y, z = vertex
            self._add_vertex_data(x, y, z)

        for face in mesh.faces:
            if face is None:  # Avoid canceled faces
                continue
            v1, v2, v3 = face  # Assume faces to be triangular
            self._add_triangle(v1, v2, v3)
