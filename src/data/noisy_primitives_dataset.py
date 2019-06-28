import random
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from framework.event_delegator import EventDelegator
from data.modifiers_encoder import ModifiersEncoder
from graphics.render.render_engine import RenderEngine
from graphics.render.renderable_mesh import RenderableMesh
from graphics.modifiers.translate_vertex import TranslateVertexModifier
from graphics.modifiers.translate_edge import TranslateEdgeModifier
from graphics.modifiers.translate_face import TranslateFaceModifier
from graphics.modifiers.split_edge import SplitEdgeModifier
from graphics.modifiers.split_vertex import SplitVertexModifier
from graphics.modifiers.contract_vertex_pair import ContractVertexPairModifier
from graphics.modifiers.create_primitive import CreatePrimitiveModifier
from graphics.modifiers.finalize_model import FinalizeModelModifier
from graphics.enums.primitive_enum import PrimitiveEnum


class NoisyPrimitivesDataset(Dataset):

    def __init__(self,
                 render_engine: RenderEngine,
                 size=5000,
                 cache=False,
                 min_modifier_steps=1,
                 max_modifier_steps=2,
                 min_pertubration=-0.5,
                 max_pertubration=0.5,
                 modifiers_pool=None):
        self.render_engine = render_engine  # Used to generate rendered images for conceived mesh
        self.size = size    # Data is generated on the fly, define how many entries to create
        self.use_cache = cache  # Whether generated data should be kept for future __getitem__() calls.
        self.cache = {}
        self.min_modifier_steps = min_modifier_steps
        self.max_modifier_steps = max_modifier_steps
        self.min_pertubration = min_pertubration
        self.max_pertubration = max_pertubration

        chosen_modifier_names = modifiers_pool or ['TranslateVertexModifier',
                                                   'TranslateEdgeModifier',
                                                   'TranslateFaceModifier',
                                                   'SplitEdgeModifier',
                                                   'SplitVertexModifier',
                                                   'ContractVertexPairModifier']
        class_name_to_class = lambda class_name: getattr(sys.modules[__name__], class_name)
        self.modfiers_pool = [class_name_to_class(modifier) for modifier in chosen_modifier_names]
        self.modifiers_encoder = ModifiersEncoder()

        # Public Events
        self.on_pre_modifier_execution = EventDelegator()
        self.on_post_modifier_execution = EventDelegator()

    def _choose_random_vertex(self, mesh):
        v_id = random.randint(0, len(mesh.vertices) - 1)
        while mesh.edges[v_id] is None:  # Make sure the vertex is not canceled
            v_id = random.randint(0, len(mesh.vertices) - 1)
        return v_id

    def _choose_random_edge(self, mesh):
        e_id = random.randint(0, len(mesh.edges) - 1)
        while mesh.edges[e_id] is None:  # Make sure the edge is not canceled
            e_id = random.randint(0, len(mesh.edges) - 1)
        return e_id

    def _choose_random_face(self, mesh):
        f_id = random.randint(0, len(mesh.faces) - 1)
        while mesh.faces[f_id] is None:  # Make sure the face is not canceled
            f_id = random.randint(0, len(mesh.faces) - 1)
        return f_id

    def _sample_uniform_perturbration(self):
        tx = random.uniform(self.min_pertubration, self.max_pertubration)
        ty = random.uniform(self.min_pertubration, self.max_pertubration)
        tz = random.uniform(self.min_pertubration, self.max_pertubration)
        return tx, ty, tz

    def _generate_modifier(self, modifier_class, mesh):

        if modifier_class == TranslateVertexModifier:
            v_id = self._choose_random_vertex(mesh)
            tx, ty, tz = self._sample_uniform_perturbration()
            modifier = TranslateVertexModifier(mesh=mesh,
                                               v_id=v_id,
                                               tx=tx, ty=ty, tz=tz)
        elif modifier_class == TranslateEdgeModifier:
            e_id = self._choose_random_edge(mesh)
            tx, ty, tz = self._sample_uniform_perturbration()
            modifier = TranslateEdgeModifier(mesh=mesh,
                                             e_id=e_id,
                                             tx=tx, ty=ty, tz=tz)
        elif modifier_class == TranslateFaceModifier:
            f_id = self._choose_random_face(mesh)
            tx, ty, tz = self._sample_uniform_perturbration()
            modifier = TranslateFaceModifier(mesh=mesh,
                                             f_id=f_id,
                                             tx=tx, ty=ty, tz=tz)
        elif modifier_class == SplitEdgeModifier:
            e_id = self._choose_random_edge(mesh)
            modifier = SplitEdgeModifier(mesh=mesh, e_id=e_id)
        elif modifier_class == SplitVertexModifier:
            v_id = self._choose_random_vertex(mesh)
            tx, ty, tz = self._sample_uniform_perturbration()
            modifier = SplitVertexModifier(mesh=mesh, v_id=v_id, tx=tx, ty=ty, tz=tz)
        elif modifier_class == ContractVertexPairModifier:
            e_id = self._choose_random_edge(mesh)
            v1_id, v2_id = mesh.edges[e_id]
            modifier = ContractVertexPairModifier(mesh=mesh, v1_id=v1_id, v2_id=v2_id)
        else:
            raise ValueError('Unsupported modifier class')

        return modifier

    def generate_noisy_cube(self):

        # TODO: Delete from here
        # from graphics.render.model_decomposition import parse_model_geometry
        # model = self.render_engine.load_model(path="/data/egg/lego.egg")
        # meshes = parse_model_geometry(model)
        # mesh = meshes[0]
        #
        # for v_id, v in enumerate(mesh.vertices):
        #     mesh.vertices[v_id] = tuple([c * 125 for c in v])
        # TODO: Delete until here and uncomment next..

        create_primitive_modifier = CreatePrimitiveModifier(primitive_type=PrimitiveEnum.Cube)
        mesh = create_primitive_modifier.execute()
        num_of_modifiers = random.randint(self.min_modifier_steps, self.max_modifier_steps)
        modifiers = [create_primitive_modifier]

        for _ in range(num_of_modifiers):

            modifier_class = self.modfiers_pool[random.randint(0, len(self.modfiers_pool) - 1)]
            modifier = self._generate_modifier(modifier_class, mesh)

            self.on_pre_modifier_execution.fire(modifier)
            mesh = modifier.execute()
            self.on_post_modifier_execution.fire(modifier)

            modifiers.append(modifier)

        modifier = FinalizeModelModifier(mesh)
        self.on_pre_modifier_execution.fire(modifier)
        mesh = modifier.execute()
        self.on_post_modifier_execution.fire(modifier)
        modifiers.append(modifier)

        return mesh, modifiers

    def encode_modifiers(self, modifiers):

        encodings = []
        max_len = 512  # TODO: Or - alter by config

        for modifier in modifiers:
            encoded_modifier = self.modifiers_encoder.encode(modifier)
            encodings.append(encoded_modifier)
            max_len = max(max_len, len(encoded_modifier))

        # Pad all encoding with zeros to ensure they're of size max_len
        encodings = list(map(lambda e: e if len(e) == max_len else F.pad(e, (0, max_len-len(e))), encodings))

        return torch.stack(encodings, dim=0)

    def to_torch_tensor(self, imgs):
        torch_tensors = [torch.from_numpy(np.ascontiguousarray(img)) for img in imgs]
        return torch.stack(torch_tensors, dim=0)

    def normalize_imgs(self, imgs):
        normalized_imgs = imgs.permute(dims=(0, 3, 1, 2))  # Return color channel before H,W; to comply with CNN format
        normalized_imgs = normalized_imgs.float()
        return normalized_imgs

    def __getitem__(self, index):

        if index in self.cache:
            return self.cache[index]

        mesh, modifiers = self.generate_noisy_cube()
        self.render_engine.clear_renderables()
        renderable_mesh = RenderableMesh(mesh)
        self.render_engine.add_renderable(renderable_mesh)
        left_img = self.render_engine.get_camera_image(camera_name='left', requested_format='RGB')
        top_img = self.render_engine.get_camera_image(camera_name='top', requested_format='RGB')
        front_img = self.render_engine.get_camera_image(camera_name='front', requested_format='RGB')

        rendered_imgs_tensors = self.to_torch_tensor(imgs=(left_img, top_img, front_img))
        rendered_imgs_tensors = self.normalize_imgs(rendered_imgs_tensors)
        encoded_modifiers = self.encode_modifiers(modifiers)

        if self.use_cache:
            self.cache[index] = (rendered_imgs_tensors, encoded_modifiers)

        return rendered_imgs_tensors, encoded_modifiers

    def __len__(self):
        return self.size

    def __str__(self):
        return 'NoisyPrimitivesDataset'

    def summary(self):
        description = {
            'Type': 'NoisyPrimitivesDataset',
            'Size': self.size,
            'Modifiers': [entry.__name__ for entry in self.modfiers_pool],
            'Parameters': {
                'min_modifier_steps': self.min_modifier_steps,
                'max_modifier_steps': self.max_modifier_steps,
                'min_pertubration': self.min_pertubration,
                'max_pertubration': self.max_pertubration
            }
        }
        return description
