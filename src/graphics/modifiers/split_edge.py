from graphics.modifiers.abstract_modifier import AbstractModifier


class SplitEdgeModifier(AbstractModifier):

    def __init__(self, mesh, e_id):
        super().__init__()
        self.mesh = mesh
        self.e_id = e_id

    def execute(self):

        edge_to_split = self.mesh.edges[self.e_id]
        v0, v1 = edge_to_split

        for face in self.mesh.edges_to_faces[edge_to_split]:
            pass    # TODO:

        return self.mesh

    def undo(self):
        for v_id in self.mesh.edges[self.e_id]:
            single_vertex_data = self.mesh.vertices[v_id]
            updated_vertex = (
                single_vertex_data[0] - self.tx,
                single_vertex_data[1] - self.ty,
                single_vertex_data[2] - self.tz,
            )

            self.mesh.vertices[v_id] = updated_vertex

        return self.mesh
