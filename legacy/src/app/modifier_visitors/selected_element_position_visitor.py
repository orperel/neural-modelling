from src.framework.abstract_modifier_visitor import AbstractModifierVisitor, visitor
from src.graphics.modifiers import TranslateVertexModifier
from src.graphics.modifiers import TranslateEdgeModifier
from src.graphics.modifiers import TranslateFaceModifier
from src.graphics.modifiers import SplitEdgeModifier
from src.graphics.modifiers import SplitVertexModifier
from src.graphics.modifiers.contract_vertex_pair import ContractVertexPairModifier
from src.graphics.modifiers.create_primitive import CreatePrimitiveModifier
from src.graphics.modifiers import FinalizeModelModifier


class SelectedElementPositionVisitor(AbstractModifierVisitor):
    """
    [(x,y,z)] position of vertices of affected element.
    Note:
    - 1 vertex if the element is a vertex.
    - 2 vertices if the element is an edge.
    - 3 vertices if the element is a face.
    """

    @staticmethod
    def _flatten_list(data):
        """ Flattens list of iterables """
        return list(sum(data, ()))

    @visitor(CreatePrimitiveModifier)
    def visit(self, modifier) -> list:
        return []

    @visitor(FinalizeModelModifier)
    def visit(self, modifier) -> list:
        return []

    @visitor(TranslateVertexModifier)
    def visit(self, modifier) -> list:
        vertex_data = modifier.vertex_data
        return list(vertex_data)

    @visitor(TranslateEdgeModifier)
    def visit(self, modifier) -> list:
        edge_data = modifier.edge_vdata
        return self._flatten_list(edge_data)

    @visitor(TranslateFaceModifier)
    def visit(self, modifier) -> list:
        face_data = modifier.face_vdata
        return self._flatten_list(face_data)

    @visitor(SplitEdgeModifier)
    def visit(self, modifier) -> list:
        edge_data = modifier.edge
        return self._flatten_list(edge_data)

    @visitor(SplitVertexModifier)
    def visit(self, modifier) -> list:
        v_data = modifier.v_data
        return self._flatten_list(v_data)

    @visitor(ContractVertexPairModifier)
    def visit(self, modifier) -> list:
        edge_data = (modifier.v1, modifier.v2)
        return self._flatten_list(edge_data)
