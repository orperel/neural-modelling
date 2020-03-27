from src.framework.abstract_modifier_visitor import AbstractModifierVisitor, visitor
from src.graphics.enums.modifier_enum import ModifierEnum
from src.graphics.modifiers import TranslateVertexModifier
from src.graphics.modifiers import TranslateEdgeModifier
from src.graphics.modifiers import TranslateFaceModifier
from src.graphics.modifiers import SplitEdgeModifier
from src.graphics.modifiers import SplitVertexModifier
from src.graphics.modifiers.contract_vertex_pair import ContractVertexPairModifier
from src.graphics.modifiers.create_primitive import CreatePrimitiveModifier
from src.graphics.modifiers import FinalizeModelModifier


class ModifierIDVisitor(AbstractModifierVisitor):
    """
    Unique ID for each modifier class
    """
    @staticmethod
    def max_id():
        all_ids = [modifier.value for modifier in ModifierEnum]
        return max(all_ids)

    @visitor(CreatePrimitiveModifier)
    def visit(self, modifier) -> int:
        return ModifierEnum.CreatePrimitiveModifier.value

    @visitor(FinalizeModelModifier)
    def visit(self, modifier) -> int:
        return ModifierEnum.FinalizeModelModifier.value

    @visitor(TranslateVertexModifier)
    def visit(self, modifier) -> int:
        return ModifierEnum.TranslateVertexModifier.value

    @visitor(TranslateEdgeModifier)
    def visit(self, modifier) -> int:
        return ModifierEnum.TranslateEdgeModifier.value

    @visitor(TranslateFaceModifier)
    def visit(self, modifier) -> int:
        return ModifierEnum.TranslateFaceModifier.value

    @visitor(SplitEdgeModifier)
    def visit(self, modifier) -> int:
        return ModifierEnum.SplitEdgeModifier.value

    @visitor(SplitVertexModifier)
    def visit(self, modifier) -> int:
        return ModifierEnum.SplitVertexModifier.value

    @visitor(ContractVertexPairModifier)
    def visit(self, modifier) -> int:
        return ModifierEnum.ContractVertexPairModifier.value
