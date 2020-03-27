import numpy as np
from src.graphics.modifiers.abstract_modifier import AbstractModifier


class OpenmeshTranslateVertexModifier(AbstractModifier):

    def __init__(self, mesh, v_handle, tx, ty, tz):
        super().__init__(mesh)
        self.v_handle = v_handle
        self.tx = tx
        self.ty = ty
        self.tz = tz

    def execute(self):
        updated_location = self.mesh.point(self.v_handle) + np.array((self.tx, self.ty, self.tz))
        self.mesh.set_point(self.v_handle, updated_location)

        return self.mesh

    def undo(self):
        updated_location = self.mesh.point(self.v_handle) - np.array((self.tx, self.ty, self.tz))
        self.mesh.set_point(self.v_handle, updated_location)

        return self.mesh

