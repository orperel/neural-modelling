from src.framework.abstract_modifier_visitor import AbstractModifierVisitor, visitor
from src.graphics.enums.element_type_enum import ElementType
from src.graphics.modifiers import TranslateVertexModifier
from src.graphics.modifiers import TranslateEdgeModifier
from src.graphics.modifiers import TranslateFaceModifier
from src.graphics.modifiers import SplitEdgeModifier
from src.graphics.modifiers import SplitVertexModifier
from src.graphics.modifiers.contract_vertex_pair import ContractVertexPairModifier
from src.graphics.modifiers.create_primitive import CreatePrimitiveModifier
from src.graphics.modifiers import FinalizeModelModifier


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
