from src.graphics.render.abstract_renderable import AbstractRenderable


class RenderableSelection(AbstractRenderable):

    def __init__(self, mesh, selection_color=None, alpha=1.0):

        super().__init__(selection_color, alpha)
        self.mesh = mesh

        self.selected_vertices = []
        self.selected_edges = []
        self.selected_faces = []

        self.vid_mapping = {}   # For mapping from mesh vid space to selection vid space, to avoid repetition

    def select_vertex(self, v_id):
        self.selected_vertices.append(v_id)

    def select_edge(self, e_id):
        edges = self.mesh.edges
        edge = edges[e_id]
        self.selected_edges.append(edge)

    def select_face(self, f_id):
        faces = self.mesh.faces
        face = faces[f_id]
        self.selected_faces.append(face)

    def _allocate_vertex_data_if_needed(self, original_v_id):
        vertices = self.mesh.vertices
        vid_mapping = self.vid_mapping

        if original_v_id not in vid_mapping:
            x, y, z = vertices[original_v_id]
            super()._add_vertex_data(x, y, z)
            vid_mapping[original_v_id] = len(vid_mapping)

        return vid_mapping[original_v_id]

    def render(self):

        for v_id in self.selected_vertices:
            next_vid = self._allocate_vertex_data_if_needed(v_id)
            super()._add_point(next_vid)
        for edge in self.selected_edges:
            v1, v2 = edge
            next_vid1 = self._allocate_vertex_data_if_needed(v1)
            next_vid2 = self._allocate_vertex_data_if_needed(v2)
            super()._add_edge(next_vid1, next_vid2)
        for face in self.selected_faces:
            v1, v2, v3 = face
            next_vid1 = self._allocate_vertex_data_if_needed(v1)
            next_vid2 = self._allocate_vertex_data_if_needed(v2)
            next_vid3 = self._allocate_vertex_data_if_needed(v3)
            super()._add_triangle(next_vid1, next_vid2, next_vid3)

        super().render()

    def clear(self):
        self.selected_vertices = []
        self.selected_edges = []
        self.selected_faces = []
        self.vid_mapping = {}
