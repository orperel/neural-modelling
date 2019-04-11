from abc import abstractmethod
from framework.event_handler import EventHandler
from app.common.colors_palette import Color
from graphics.render.renderable_mesh import RenderableMesh
from graphics.render.renderable_selection import RenderableSelection
from graphics.render.render_engine import RenderEngine
from graphics.modifiers.abstract_modifier import AbstractModifier
from app.modifier_visitors.affected_elements_visitor import AffectedElementsVisitor


class VisualizeModifierEventHandler(EventHandler):

    def __init__(self, render_engine: RenderEngine):
        super().__init__()
        self.render_engine = render_engine

    @staticmethod
    def _populate_renderable_from_selection(selected_elements: dict,
                                            renderable_selection: RenderableSelection):

        for v_id in selected_elements['vertices']:
            renderable_selection.select_vertex(v_id)
        for e_id in selected_elements['edges']:
            renderable_selection.select_edge(e_id)
        for f_id in selected_elements['faces']:
            renderable_selection.select_face(f_id)

    def handle(self, modifier: AbstractModifier):

        affected_elements_category = self._get_affected_elements_category()
        selection_color = self._get_selection_color().value

        visitor = AffectedElementsVisitor()
        selected_elements = visitor.visit(modifier)[affected_elements_category]
        renderable_selection = RenderableSelection(mesh=modifier.mesh, selection_color=selection_color)
        self._populate_renderable_from_selection(selected_elements, renderable_selection)

        renderable_obj = RenderableMesh(modifier.mesh, alpha=0.5)

        self.render_engine.clear_renderables()
        self.render_engine.add_renderable(renderable_obj)
        self.render_engine.add_renderable(renderable_selection)
        self.render_engine.run_for(seconds=2)

    @staticmethod
    @abstractmethod
    def _get_affected_elements_category() -> str:
        pass

    @staticmethod
    @abstractmethod
    def _get_selection_color() -> Color:
        pass
