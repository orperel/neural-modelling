from graphics.modifiers.abstract_modifier import AbstractModifier


class TranslateFaceModifier(AbstractModifier):

    def __init__(self, mesh, f_id, tx, ty, tz):
        super().__init__()
        self.mesh = mesh
        self.f_id = f_id
        self.tx = tx
        self.ty = ty
        self.tz = tz

    def execute(self):
        for v_id in self.mesh.faces[self.f_id]:
            single_vertex_data = self.mesh.vertices[v_id]
            updated_vertex = (
                single_vertex_data[0] + self.tx,
                single_vertex_data[1] + self.ty,
                single_vertex_data[2] + self.tz,
            )

            self.mesh.vertices[v_id] = updated_vertex

        return self.mesh

    def undo(self):
        for v_id in self.mesh.faces[self.f_id]:
            single_vertex_data = self.mesh.vertices[v_id]
            updated_vertex = (
                single_vertex_data[0] - self.tx,
                single_vertex_data[1] - self.ty,
                single_vertex_data[2] - self.tz,
            )

            self.mesh.vertices[v_id] = updated_vertex

        return self.mesh
