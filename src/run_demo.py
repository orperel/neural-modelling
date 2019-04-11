import torch
import yaml
import cv2
import sys
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
from model_loader import load_model
from graphics.render.model_decomposition import parse_model_geometry


def load_config():
    with open('config.yaml', 'r') as yaml_file:
        config_file = yaml.load(yaml_file)
    return config_file


def load_dataset(config, engine):
    dataset_config = config['TRAIN']['DATASET']

    if dataset_config['TYPE'] == 'noisy_primitives':
        class_name_to_class = lambda class_name: getattr(sys.modules[__name__], class_name)
        modifiers_pool = [class_name_to_class(modifier) for modifier in dataset_config['MODIFIERS']]
        dataset = NoisyPrimitivesDataset(render_engine=engine,
                                         size=dataset_config['SIZE'],
                                         cache=dataset_config['CACHE'],
                                         min_modifier_steps=dataset_config['MIN_MODIFIER_STEPS'],
                                         max_modifier_steps=dataset_config['MAX_MODIFIER_STEPS'],
                                         modifiers_pool=modifiers_pool,
                                         min_pertubration=dataset_config['MIN_PERTUBRATION'],
                                         max_pertubration=dataset_config['MAX_PERTUBRATION']
                                         )
    else:
        raise ValueError('Unknown dataset type encountered in config: %r' % dataset_config['TYPE'])

    return dataset

def handle_model(engine, model_path="/data/egg/lego.egg"):
    model = engine.load_model(path=model_path)
    meshes = parse_model_geometry(model)

    mesh = meshes[0]
    renderable_obj = RenderableMesh(mesh)
    engine.clear_renderables()
    engine.add_renderable(renderable_obj)
    engine.start_rendering_loop()

config = load_config()
engine = RenderEngine()
interactive = InteractiveWidget(engine)
handle_model(engine)
dataset = load_dataset(config, engine)
model = load_model(config)
dataset.on_pre_modifier_execution += VisualizePreModifierEventHandler(engine)
dataset.on_post_modifier_execution += VisualizePostModifierEventHandler(engine)

rendered_triplet, modifiers, cube = dataset.__getitem__(0)
left_img, top_img, front_img = rendered_triplet
cv2.imshow('Left Render', left_img)
cv2.imshow('Top Render', top_img)
cv2.imshow('Front Render', front_img)
print(modifiers)

renderable_obj = RenderableMesh(cube)
engine.clear_renderables()
engine.add_renderable(renderable_obj)
engine.start_rendering_loop()
