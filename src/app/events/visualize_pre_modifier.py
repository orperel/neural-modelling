from app.events.abstract_visualize_modifier import VisualizeModifierEventHandler
from app.common.colors_palette import Color


class VisualizePreModifierEventHandler(VisualizeModifierEventHandler):

    def __init__(self, render_engine):
        self.render_engine = render_engine

    @staticmethod
    def _get_affected_elements_category() -> str:
        return 'pre_modification'

    @staticmethod
    def _get_selection_color() -> Color:
        return Color.PINK

