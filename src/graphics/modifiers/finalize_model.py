from graphics.modifiers.abstract_modifier import AbstractModifier


class FinalizeModelModifier(AbstractModifier):

    def __init__(self, mesh):
        super().__init__(mesh)

    def execute(self):
        return self.mesh

    def undo(self):
        return self.mesh
