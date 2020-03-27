from src.app.events.abstract_visualize_modifier import VisualizeModifierEventHandler
from src.app.common import Color


class VisualizePreModifierEventHandler(VisualizeModifierEventHandler):

    def __init__(self, render_engine):
        super().__init__(render_engine)

    @staticmethod
    def _get_affected_elements_category() -> str:
        return 'pre_modification'

    @staticmethod
    def _get_selection_color() -> Color:
        return Color.PINK

