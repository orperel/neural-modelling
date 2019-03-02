import random
from graphics.units.procedural_primitives import ProceduralPrimitives
from graphics.modifiers.translate_vertex import TranslateVertexModifier


class NoisyPrimitivesDataset:

    def __init__(self,
                 min_modifier_steps=3,
                 max_modifier_steps=15,
                 modifiers_pool=None):
        self.min_modifier_steps = min_modifier_steps
        self.max_modifier_steps = max_modifier_steps
        self.modfiers_pool = modifiers_pool or [TranslateVertexModifier]

    @staticmethod
    def _generate_modifier(modifier_class, mesh):

        if modifier_class == TranslateVertexModifier:
            v_id = random.randint(0, len(mesh.vertices) - 1)
            tx = random.uniform(0.0, 0.5)
            ty = random.uniform(0.0, 0.5)
            tz = random.uniform(0.0, 0.5)

            modifier = TranslateVertexModifier(mesh=mesh,
                                               v_id=v_id,
                                               tx=tx, ty=ty, tz=tz)
        else:
            raise ValueError('Unsupported modifier class')

        return modifier

    def generate_noisy_cube(self):

        mesh = ProceduralPrimitives.generate_cube()
        num_of_modifiers = random.randint(self.min_modifier_steps, self.max_modifier_steps)

        for _ in range(num_of_modifiers):

            modifier_class = self.modfiers_pool[random.randint(0, len(self.modfiers_pool) - 1)]
            modifier = self._generate_modifier(modifier_class, mesh)
            mesh = modifier.execute()

        return mesh
