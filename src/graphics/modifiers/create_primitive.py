from graphics.modifiers.abstract_modifier import AbstractModifier
from graphics.enums.primitive_enum import PrimitiveEnum
from graphics.geometry.procedural_primitives import ProceduralPrimitives


class CreatePrimitiveModifier(AbstractModifier):

    def __init__(self, primitive_type: PrimitiveEnum):
        super().__init__(mesh=None)
        self.primitive_type = primitive_type
        self.initial_vertex_ids = None

    def execute(self):
        if self.primitive_type == PrimitiveEnum.Cube:
            self.mesh = ProceduralPrimitives.generate_cube()
        else:
            raise ValueError("CreatePrimitiveModifier :: Unsupported primitive type %r" % self.primitive_type)
        
        self.initial_vertex_ids = list(range(len(self.mesh.vertices)))  # For visualizations
        return self.mesh

    def undo(self):
        self.mesh = None

