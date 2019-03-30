from app.events.abstract_visualize_modifier import VisualizeModifierEventHandler
from app.common.colors_palette import Color


class VisualizePostModifierEventHandler(VisualizeModifierEventHandler):

    def __init__(self, render_engine):
        self.render_engine = render_engine

    @staticmethod
    def _get_affected_elements_category() -> str:
        return 'post_modification'

    @staticmethod
    def _get_selection_color() -> Color:
        return Color.GREEN

