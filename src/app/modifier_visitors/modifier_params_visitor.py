from framework.abstract_modifier_visitor import AbstractModifierVisitor, visitor
from graphics.modifiers.translate_vertex import TranslateVertexModifier
from graphics.modifiers.translate_edge import TranslateEdgeModifier
from graphics.modifiers.translate_face import TranslateFaceModifier
from graphics.modifiers.split_edge import SplitEdgeModifier
from graphics.modifiers.split_vertex import SplitVertexModifier
from graphics.modifiers.contract_vertex_pair import ContractVertexPairModifier
from graphics.modifiers.create_primitive import CreatePrimitiveModifier
from graphics.modifiers.finalize_model import FinalizeModelModifier


class ModifierParamsVisitor(AbstractModifierVisitor):
    """
    List of parameters used by this modifier.
    For example: A translation modifier will return the translation vector here.
    If the modifier uses no parameters, an empty list is returned.
    """

    @visitor(CreatePrimitiveModifier)
    def visit(self, modifier) -> list:
        return [modifier.primitive_type.value]

    @visitor(FinalizeModelModifier)
    def visit(self, modifier) -> list:
        return []

    @visitor(TranslateVertexModifier)
    def visit(self, modifier) -> list:
        return [modifier.tx, modifier.ty, modifier.tz]

    @visitor(TranslateEdgeModifier)
    def visit(self, modifier) -> list:
        return [modifier.tx, modifier.ty, modifier.tz]

    @visitor(TranslateFaceModifier)
    def visit(self, modifier) -> list:
        return [modifier.tx, modifier.ty, modifier.tz]

    @visitor(SplitEdgeModifier)
    def visit(self, modifier) -> list:
        return []

    @visitor(SplitVertexModifier)
    def visit(self, modifier) -> list:
        return [modifier.tx, modifier.ty, modifier.tz]

    @visitor(ContractVertexPairModifier)
    def visit(self, modifier) -> list:
        return []
