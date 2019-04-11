from enum import IntEnum
from framework.abstract_modifier_visitor import AbstractModifierVisitor, visitor
from graphics.modifiers.split_edge import SplitEdgeModifier
from graphics.modifiers.translate_vertex import TranslateVertexModifier
from graphics.modifiers.translate_edge import TranslateEdgeModifier
from graphics.modifiers.translate_face import TranslateFaceModifier


class ElementType(IntEnum):
    VERTEX = 0
    EDGE = 1
    FACE = 2


class SelectedElementTypeVisitor(AbstractModifierVisitor):

    @visitor(TranslateVertexModifier)
    def visit(self, modifier) -> ElementType:
        return ElementType.VERTEX

    @visitor(TranslateEdgeModifier)
    def visit(self, modifier) -> ElementType:
        return ElementType.EDGE

    @visitor(TranslateFaceModifier)
    def visit(self, modifier) -> ElementType:
        return ElementType.FACE

    @visitor(SplitEdgeModifier)
    def visit(self, modifier) -> ElementType:
        return ElementType.EDGE
