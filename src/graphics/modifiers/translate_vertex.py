from graphics.modifiers.abstract_modifier import AbstractModifier


class TranslateVertexModifier(AbstractModifier):

    def __init__(self, mesh, v_id, tx, ty, tz):
        self.mesh = mesh
        self.v_id = v_id
        self.tx = tx
        self.ty = ty
        self.tz = tz

    def execute(self):
        single_vertex_data = self.mesh.vertices[self.v_id]
        updated_vertex = (
            single_vertex_data[0] + self.tx,
            single_vertex_data[1] + self.ty,
            single_vertex_data[2] + self.tz,
        )

        self.mesh.vertices[self.v_id] = updated_vertex

        return self.mesh

    def undo(self):
        single_vertex_data = self.mesh.vertices[self.v_id]
        updated_vertex = (
            single_vertex_data[0] - self.tx,
            single_vertex_data[1] - self.ty,
            single_vertex_data[2] - self.tz,
        )

        self.mesh.vertices[self.v_id] = updated_vertex

        return self.mesh
