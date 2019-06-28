from framework.abstract_modifier_visitor import AbstractModifierVisitor, visitor
from graphics.enums.modifier_enum import ModifierEnum
from graphics.modifiers.translate_vertex import TranslateVertexModifier
from graphics.modifiers.translate_edge import TranslateEdgeModifier
from graphics.modifiers.translate_face import TranslateFaceModifier
from graphics.modifiers.split_edge import SplitEdgeModifier
from graphics.modifiers.split_vertex import SplitVertexModifier
from graphics.modifiers.contract_vertex_pair import ContractVertexPairModifier
from graphics.modifiers.create_primitive import CreatePrimitiveModifier
from graphics.modifiers.finalize_model import FinalizeModelModifier


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
