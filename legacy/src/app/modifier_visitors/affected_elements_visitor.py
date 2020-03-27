from src.framework.abstract_modifier_visitor import AbstractModifierVisitor, visitor
from src.graphics.modifiers import TranslateVertexModifier
from src.graphics.modifiers import TranslateEdgeModifier
from src.graphics.modifiers import TranslateFaceModifier
from src.graphics.modifiers import SplitEdgeModifier
from src.graphics.modifiers import SplitVertexModifier
from src.graphics.modifiers.contract_vertex_pair import ContractVertexPairModifier
from src.graphics.modifiers.create_primitive import CreatePrimitiveModifier
from src.graphics.modifiers import FinalizeModelModifier

from src.graphics.modifiers.openmesh import OpenmeshTranslateVertexModifier
from src.graphics.modifiers.openmesh import OpenMeshSplitVertexModifier


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
                'vertices': [],
                'edges': [*modifier.new_edges],
                'faces': []
            }
        }

    @visitor(SplitVertexModifier)
    def visit(self, modifier) -> dict:
        return {
            'pre_modification': {
                'vertices': [modifier.v_id],
                'edges': [],
                'faces': []
            },
            'post_modification': {
                'vertices': [modifier.new_v1_id, modifier.new_v1_id],
                'edges': [],
                'faces': []
            }
        }

    @visitor(ContractVertexPairModifier)
    def visit(self, modifier) -> dict:
        return {
            'pre_modification': {
                'vertices': [modifier.v1_id, modifier.v2_id],
                'edges': [],
                'faces': []
            },
            'post_modification': {
                'vertices': [modifier.v_id],
                'edges': [],
                'faces': []
            }
        }

    @visitor(CreatePrimitiveModifier)
    def visit(self, modifier) -> dict:
        return {
            'pre_modification': {
                'vertices': [],
                'edges': [],
                'faces': []
            },
            'post_modification': {
                'vertices': [*modifier.initial_vertex_ids],
                'edges': [],
                'faces': []
            }
        }

    @visitor(FinalizeModelModifier)
    def visit(self, modifier) -> dict:
        return {
            'pre_modification': {
                'vertices': [],
                'edges': [],
                'faces': []
            },
            'post_modification': {
                'vertices': [],
                'edges': [],
                'faces': []
            }
        }

    @visitor(OpenmeshTranslateVertexModifier)
    def visit(self, modifier) -> dict:
        return {
            'pre_modification': {
                'vertices': [modifier.v_handle],
                'edges': [],
                'faces': []
            },
            'post_modification': {
                'vertices': [modifier.v_handle],
                'edges': [],
                'faces': []
            }
        }

    @visitor(OpenMeshSplitVertexModifier)
    def visit(self, modifier) -> dict:
        return {
            'pre_modification': {
                'vertices': [modifier.v1_idx],
                'edges': [],
                'faces': []
            },
            'post_modification': {
                'vertices': [],
                'edges': [modifier.halfedge_handle],
                'faces': []
            }
        }
