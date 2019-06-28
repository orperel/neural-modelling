from framework.abstract_modifier_visitor import AbstractModifierVisitor, visitor
from graphics.enums.element_type_enum import ElementType
from graphics.modifiers.translate_vertex import TranslateVertexModifier
from graphics.modifiers.translate_edge import TranslateEdgeModifier
from graphics.modifiers.translate_face import TranslateFaceModifier
from graphics.modifiers.split_edge import SplitEdgeModifier
from graphics.modifiers.split_vertex import SplitVertexModifier
from graphics.modifiers.contract_vertex_pair import ContractVertexPairModifier
from graphics.modifiers.create_primitive import CreatePrimitiveModifier
from graphics.modifiers.finalize_model import FinalizeModelModifier


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

    @visitor(SplitVertexModifier)
    def visit(self, modifier) -> ElementType:
        return ElementType.VERTEX

    @visitor(ContractVertexPairModifier)
    def visit(self, modifier) -> ElementType:
        return ElementType.EDGE

    @visitor(CreatePrimitiveModifier)
    def visit(self, modifier) -> ElementType:
        return ElementType.ELEMENT

    @visitor(FinalizeModelModifier)
    def visit(self, modifier) -> ElementType:
        return ElementType.ELEMENT
