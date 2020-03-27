from src.framework.abstract_modifier_visitor import AbstractModifierVisitor, visitor
from src.graphics.modifiers import TranslateVertexModifier
from src.graphics.modifiers import TranslateEdgeModifier
from src.graphics.modifiers import TranslateFaceModifier
from src.graphics.modifiers import SplitEdgeModifier
from src.graphics.modifiers import SplitVertexModifier
from src.graphics.modifiers.contract_vertex_pair import ContractVertexPairModifier
from src.graphics.modifiers.create_primitive import CreatePrimitiveModifier
from src.graphics.modifiers import FinalizeModelModifier


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
