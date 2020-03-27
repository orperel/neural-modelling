import yaml
from src.graphics.render.interactive import InteractiveWidget
from src.graphics.render.render_engine import RenderEngine
from src.graphics import RenderableMesh
from src.data.noisy_primitives_dataset import NoisyPrimitivesDataset
from src.app import VisualizePreModifierEventHandler
from src.app.events.visualize_post_modifier import VisualizePostModifierEventHandler
from src.graphics import parse_model_geometry
import torch
import random
import numpy as np


def load_config():
    with open('configs/debug_config.yaml', 'r') as yaml_file:
        config_file = yaml.load(yaml_file)
    return config_file


def set_seed(config):
    if config['SEED'] is not None:
        seed = config['SEED']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if config['CUDA']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def load_dataset(config, engine):
    dataset_config = config['TRAIN']['DATASET']

    dataset = NoisyPrimitivesDataset(render_engine=engine,
                                     size=dataset_config['SIZE'],
                                     cache=dataset_config['CACHE'],
                                     min_modifier_steps=dataset_config['MIN_MODIFIER_STEPS'],
                                     max_modifier_steps=dataset_config['MAX_MODIFIER_STEPS'],
                                     modifiers_pool=dataset_config['MODIFIERS'],
                                     min_pertubration=dataset_config['MIN_PERTUBRATION'],
                                     max_pertubration=dataset_config['MAX_PERTUBRATION']
                                     )

    dataset.on_pre_modifier_execution += VisualizePreModifierEventHandler(engine)
    dataset.on_post_modifier_execution += VisualizePostModifierEventHandler(engine)

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
set_seed(config)
engine = RenderEngine(config['RENDERING'])
interactive = InteractiveWidget(engine)
# handle_model(engine)
dataset = load_dataset(config, engine)
# model = load_model(config)

rendered_triplet, modifiers, cube = dataset.__getitem__(0)
left_img, top_img, front_img = rendered_triplet
# cv2.imshow('Left Render', left_img)
# cv2.imshow('Top Render', top_img)
# cv2.imshow('Front Render', front_img)
# print(modifiers)

renderable_obj = RenderableMesh(cube)
engine.clear_renderables()
engine.add_renderable(renderable_obj)
engine.start_rendering_loop()
