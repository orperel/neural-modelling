import yaml
from src.graphics.render.render_engine import RenderEngine
from src.graphics import RenderableMesh
from src.graphics import parse_model_geometry
from src.training import Train
import torch
import random
import numpy as np


def load_config():
    with open('configs/training_config.yaml', 'r') as yaml_file:
        config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
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

if config['RENDERING']['RENDERING_ENGINE_ON']:
    engine = RenderEngine(rendering_config=config['RENDERING'])
else:
    engine = None

trainer = Train(config=config, engine=engine)
trainer.train()
