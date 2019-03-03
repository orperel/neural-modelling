import random
from torch.utils.data import Dataset
from graphics.units.procedural_primitives import ProceduralPrimitives
from graphics.modifiers.translate_vertex import TranslateVertexModifier
from graphics.modifiers.translate_edge import TranslateEdgeModifier
from graphics.modifiers.translate_face import TranslateFaceModifier


class NoisyPrimitivesDataset(Dataset):

    def __init__(self,
                 size=5000,
                 min_modifier_steps=1,
                 max_modifier_steps=2,
                 min_pertubration=-0.5,
                 max_pertubration=0.5,
                 modifiers_pool=None):
        self.size = size    # Data is generated on the fly, define how many entries to create
        self.min_modifier_steps = min_modifier_steps
        self.max_modifier_steps = max_modifier_steps
        self.min_pertubration = min_pertubration
        self.max_pertubration = max_pertubration
        self.modfiers_pool = modifiers_pool or [TranslateVertexModifier, TranslateEdgeModifier, TranslateFaceModifier]

    def _generate_modifier(self, modifier_class, mesh):

        if modifier_class == TranslateVertexModifier:
            v_id = random.randint(0, len(mesh.vertices) - 1)
            tx = random.uniform(self.min_pertubration, self.max_pertubration)
            ty = random.uniform(self.min_pertubration, self.max_pertubration)
            tz = random.uniform(self.min_pertubration, self.max_pertubration)

            modifier = TranslateVertexModifier(mesh=mesh,
                                               v_id=v_id,
                                               tx=tx, ty=ty, tz=tz)
        elif modifier_class == TranslateEdgeModifier:
            e_id = random.randint(0, len(mesh.edges) - 1)
            tx = random.uniform(self.min_pertubration, self.max_pertubration)
            ty = random.uniform(self.min_pertubration, self.max_pertubration)
            tz = random.uniform(self.min_pertubration, self.max_pertubration)

            modifier = TranslateEdgeModifier(mesh=mesh,
                                             e_id=e_id,
                                             tx=tx, ty=ty, tz=tz)
        elif modifier_class == TranslateFaceModifier:
            f_id = random.randint(0, len(mesh.faces) - 1)
            tx = random.uniform(self.min_pertubration, self.max_pertubration)
            ty = random.uniform(self.min_pertubration, self.max_pertubration)
            tz = random.uniform(self.min_pertubration, self.max_pertubration)

            modifier = TranslateFaceModifier(mesh=mesh,
                                             f_id=f_id,
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

    def __getitem__(self, index):
        return self.generate_noisy_cube()

    def __len__(self):
        return self.size
