import torch
from training.modifier_labels_decoder import ModifierLabelsDecoder
from graphics.enums.modifier_enum import ModifierEnum
from graphics.enums.element_type_enum import ElementType
from graphics.modifiers.translate_vertex import TranslateVertexModifier
from graphics.modifiers.translate_edge import TranslateEdgeModifier
from graphics.modifiers.translate_face import TranslateFaceModifier
from graphics.modifiers.split_edge import SplitEdgeModifier
from graphics.modifiers.split_vertex import SplitVertexModifier
from graphics.modifiers.contract_vertex_pair import ContractVertexPairModifier
from graphics.geometry.mesh_iterator import iterate_mesh_vertices, get_edge_from_vertex_ids, get_face_from_vertex_ids
from graphics.geometry.mesh_utils import vertex_distance
from graphics.geometry.procedural_primitives import ProceduralPrimitives


class ModifiersFactory:
    def __init__(self):
        self.labels_decoder = ModifierLabelsDecoder()
        self.mesh = ProceduralPrimitives.generate_cube()

    def from_tensor(self, encoding):

        decoded_components = self.labels_decoder.decode(encoding)

        modifier_class_id, \
        element_type_tensor, \
        element_pos_tensor, \
        element_pos_tensor_mask, \
        modifier_params, \
        modifier_params_mask = decoded_components

        modifier_class = self.modifier_class_from_code(modifier_class_id)
        element_id = self.soft_select_element_from_mesh(element_type_tensor, element_pos_tensor)
        modifier_object_params = self._gather_modifier_params_list(modifier_params, modifier_params_mask)

        modifier = modifier_class(element_id, *modifier_object_params)
        return modifier

    @staticmethod
    def modifier_class_from_code(modifier_class_id):
        """
        Convert modifier id to modifier class
        :param modifier_class_id: Modifier type id
        :return: A subclass of AbstractModifier
        """
        if modifier_class_id == ModifierEnum.TranslateVertexModifier.value:
            return TranslateVertexModifier
        elif modifier_class_id == ModifierEnum.TranslateEdgeModifier.value:
            return TranslateEdgeModifier
        elif modifier_class_id == ModifierEnum.TranslateFaceModifier.value:
            return TranslateFaceModifier
        elif modifier_class_id == ModifierEnum.SplitEdgeModifier.value:
            return SplitEdgeModifier
        elif modifier_class_id == ModifierEnum.SplitVertexModifier.value:
            return SplitVertexModifier
        elif modifier_class_id == ModifierEnum.ContractVertexPairModifier.value:
            return ContractVertexPairModifier
        else:
            raise ValueError('DecodeError: Unknown modifier class id encountered')

    def soft_select_element_from_mesh(self, element_type_tensor, element_pos_tensor):

        if element_type_tensor == ElementType.VERTEX:
            vertex_pos = element_pos_tensor
            v_id = self._soft_select_vertex_nearest_to_pos(vertex_pos)
            return v_id
        elif element_type_tensor == ElementType.EDGE:
            vertex_pos_a = element_pos_tensor[:3]
            vertex_pos_b = element_pos_tensor[3:6]
            v_id_a = self._soft_select_vertex_nearest_to_pos(vertex_pos_a)
            v_id_b = self._soft_select_vertex_nearest_to_pos(vertex_pos_b)
            e_id = get_edge_from_vertex_ids(v_id_a, v_id_b)
            return e_id
        elif element_type_tensor == ElementType.FACE:
            vertex_pos_a = element_pos_tensor[:3]
            vertex_pos_b = element_pos_tensor[3:6]
            vertex_pos_c = element_pos_tensor[6:9]
            v_id_a = self._soft_select_vertex_nearest_to_pos(vertex_pos_a)
            v_id_b = self._soft_select_vertex_nearest_to_pos(vertex_pos_b)
            v_id_c = self._soft_select_vertex_nearest_to_pos(vertex_pos_c)
            f_id = get_face_from_vertex_ids(v_id_a, v_id_b, v_id_c)
            return f_id
        else:
            raise ValueError('DecodeError: Unknown element type encountered')

    def _soft_select_vertex_nearest_to_pos(self, pos):

        selection_pos = tuple(pos.squeeze().cpu().numpy())
        selected_v_id = None
        min_distance = float("inf")

        for v_id, vertex in iterate_mesh_vertices(self.mesh):
            v_dist = vertex_distance(selection_pos, vertex)
            if v_dist < min_distance:
                min_distance = v_dist
                selected_v_id = v_id

        return selected_v_id

    def _gather_modifier_params_list(self, modifier_params, modifier_params_mask):
        params_count = torch.sum(modifier_params_mask)  # Number of ones

        params = []
        for idx in range(params_count):
            params.append(modifier_params[idx])

        return params
