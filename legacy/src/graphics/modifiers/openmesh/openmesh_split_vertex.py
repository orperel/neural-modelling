import numpy as np
from src.graphics import Mesh
from src.graphics.modifiers.abstract_modifier import AbstractModifier


class OpenMeshSplitVertexModifier(AbstractModifier):

    def __init__(self, mesh: Mesh, v1_idx, vl_idx, vr_idx, tx, ty, tz):
        super().__init__(mesh)
        self.v1_idx = v1_idx
        self.vl_idx = vl_idx
        self.vr_idx = vr_idx
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.halfedge_handle = None

    @staticmethod
    def _new_vertex_positions(v, tx, ty, tz):
        v0 = v + np.array((tx, ty, tz))
        v1 = v
        return v0, v1

    def execute(self):
        v1 = self.mesh.vertex_handle(self.v1_idx)
        vl = self.mesh.vertex_handle(self.vl_idx)
        vr = self.mesh.vertex_handle(self.vr_idx)
        v1_pos = self.mesh.point(v1)
        v0_pos, v1_pos = self._new_vertex_positions(v1_pos, self.tx, self.ty, self.tz)
        self.halfedge_handle = self.mesh.vertex_split(v0_pos, v1, vl, vr)
        return self.mesh

    def undo(self):
        self.mesh.collapse(self.halfedge_handle)
