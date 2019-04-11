from graphics.modifiers.abstract_modifier import AbstractModifier


class TranslateEdgeModifier(AbstractModifier):

    def __init__(self, mesh, e_id, tx, ty, tz):
        super().__init__(mesh)
        self.e_id = e_id
        self.tx = tx
        self.ty = ty
        self.tz = tz

    def execute(self):
        for v_id in self.mesh.edges[self.e_id]:
            single_vertex_data = self.mesh.vertices[v_id]
            updated_vertex = (
                single_vertex_data[0] + self.tx,
                single_vertex_data[1] + self.ty,
                single_vertex_data[2] + self.tz,
            )

            self.mesh.vertices[v_id] = updated_vertex

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

    def affected_elements(self) -> dict:
        return {
            'pre_modification': {
                'vertices': [],
                'edges': [self.e_id],
                'faces': []
            },
            'post_modification': {
                'vertices': [],
                'edges': [self.e_id],
                'faces': []
            }
        }