import torch
import cv2
from graphics.render.interactive import InteractiveWidget
from graphics.render.render_engine import RenderEngine
from graphics.render.renderable_mesh import RenderableMesh
from graphics.geometry.procedural_primitives import ProceduralPrimitives
from src.data.noisy_primitives_dataset import NoisyPrimitivesDataset
from graphics.modifiers.translate_vertex import TranslateVertexModifier
from graphics.modifiers.translate_edge import TranslateEdgeModifier
from graphics.modifiers.translate_face import TranslateFaceModifier
from graphics.modifiers.split_edge import SplitEdgeModifier
from app.events.visualize_pre_modifier import VisualizePreModifierEventHandler
from app.events.visualize_post_modifier import VisualizePostModifierEventHandler

engine = RenderEngine()
interactive = InteractiveWidget(engine)
dataset = NoisyPrimitivesDataset(render_engine=engine,
                                 min_modifier_steps=2, max_modifier_steps=10,
                                 modifiers_pool=[TranslateEdgeModifier, SplitEdgeModifier],
                                 min_pertubration=0.05,
                                 max_pertubration=0.5,
                                 )
# dataset.on_pre_modifier_execution += VisualizePreModifierEventHandler(engine)
# dataset.on_post_modifier_execution += VisualizePostModifierEventHandler(engine)

rendered_triplet, cube = dataset.__getitem__(0)
left_img, top_img, front_img = rendered_triplet
cv2.imshow('Left Render', left_img)
cv2.imshow('Top Render', top_img)
cv2.imshow('Front Render', front_img)

# model = engine.load_model(path="/data/egg/lego.egg")
# parse_model_geometry(model)

renderable_obj = RenderableMesh(cube)
engine.clear_renderables()
engine.add_renderable(renderable_obj)
engine.start_rendering_loop()
