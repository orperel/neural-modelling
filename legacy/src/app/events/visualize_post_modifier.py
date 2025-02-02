from src.app.events.abstract_visualize_modifier import VisualizeModifierEventHandler
from src.app.common import Color


class VisualizePostModifierEventHandler(VisualizeModifierEventHandler):

    def __init__(self, render_engine):
        super().__init__(render_engine)

    @staticmethod
    def _get_affected_elements_category() -> str:
        return 'post_modification'

    @staticmethod
    def _get_selection_color() -> Color:
        return Color.GREEN

