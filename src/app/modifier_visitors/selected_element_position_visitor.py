from framework.abstract_modifier_visitor import AbstractModifierVisitor, visitor
from graphics.modifiers.split_edge import SplitEdgeModifier
from graphics.modifiers.translate_vertex import TranslateVertexModifier
from graphics.modifiers.translate_edge import TranslateEdgeModifier
from graphics.modifiers.translate_face import TranslateFaceModifier


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
