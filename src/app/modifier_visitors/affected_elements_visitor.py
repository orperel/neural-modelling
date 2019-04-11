from framework.abstract_modifier_visitor import AbstractModifierVisitor, visitor
from graphics.modifiers.split_edge import SplitEdgeModifier
from graphics.modifiers.translate_vertex import TranslateVertexModifier
from graphics.modifiers.translate_edge import TranslateEdgeModifier
from graphics.modifiers.translate_face import TranslateFaceModifier


class AffectedElementsVisitor(AbstractModifierVisitor):
    """
    Returns dict containing ids of affected elements before and after the modifier is applied.
    For each visit implementation -
    :return:
    {
        'pre_modification': {
            'vertices': [ vid1, ...],
            'edges': [ eid1, ...],
            'faces': [ fid1, ...]
        },
        'post_modification': {
            'vertices': [ vid1, ...],
            'edges': [ eid1, ...],
            'faces': [ fid1, ...]
        }
    }
    """

    @visitor(TranslateVertexModifier)
    def visit(self, modifier) -> dict:
        return {
            'pre_modification': {
                'vertices': [modifier.v_id],
                'edges': [],
                'faces': []
            },
            'post_modification': {
                'vertices': [modifier.v_id],
                'edges': [],
                'faces': []
            }
        }

    @visitor(TranslateEdgeModifier)
    def visit(self, modifier) -> dict:
        return {
            'pre_modification': {
                'vertices': [],
                'edges': [modifier.e_id],
                'faces': []
            },
            'post_modification': {
                'vertices': [],
                'edges': [modifier.e_id],
                'faces': []
            }
        }

    @visitor(TranslateFaceModifier)
    def visit(self, modifier) -> dict:
        return {
            'pre_modification': {
                'vertices': [],
                'edges': [],
                'faces': [modifier.f_id]
            },
            'post_modification': {
                'vertices': [],
                'edges': [],
                'faces': [modifier.f_id]
            }
        }

    @visitor(SplitEdgeModifier)
    def visit(self, modifier) -> dict:
        return {
            'pre_modification': {
                'vertices': [],
                'edges': [modifier.e_id],
                'faces': []
            },
            'post_modification': {
                'vertices': [modifier.new_vid],
                'edges': modifier.new_edges,
                'faces': []
            }
        }
