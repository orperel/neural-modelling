from framework.abstract_modifier_visitor import AbstractModifierVisitor, visitor
from graphics.modifiers.split_edge import SplitEdgeModifier
from graphics.modifiers.translate_vertex import TranslateVertexModifier
from graphics.modifiers.translate_edge import TranslateEdgeModifier
from graphics.modifiers.translate_face import TranslateFaceModifier


class ModifierParamsVisitor(AbstractModifierVisitor):
    """
    List of parameters used by this modifier.
    For example: A translation modifier will return the translation vector here.
    If the modifier uses no parameters, an empty list is returned.
    """

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
