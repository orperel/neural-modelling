from enum import IntEnum
from framework.abstract_modifier_visitor import AbstractModifierVisitor, visitor
from graphics.modifiers.split_edge import SplitEdgeModifier
from graphics.modifiers.translate_vertex import TranslateVertexModifier
from graphics.modifiers.translate_edge import TranslateEdgeModifier
from graphics.modifiers.translate_face import TranslateFaceModifier


class ModifierEnum(IntEnum):
    TranslateVertexModifier = 0
    TranslateEdgeModifier = 1
    TranslateFaceModifier = 2
    SplitEdgeModifier = 3


class ModifierIDVisitor(AbstractModifierVisitor):
    """
    Unique ID for each modifier class
    """
    @staticmethod
    def max_id():
        all_ids = [modifier.value for modifier in ModifierEnum]
        return max(all_ids)

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
