import yaml
from graphics.render.render_engine import RenderEngine
from graphics.render.renderable_mesh import RenderableMesh
from graphics.render.model_decomposition import parse_model_geometry
from training.train import Train


def load_config():
    with open('config.yaml', 'r') as yaml_file:
        config_file = yaml.load(yaml_file)
    return config_file


def handle_model(engine, model_path="/data/egg/lego.egg"):
    model = engine.load_model(path=model_path)
    meshes = parse_model_geometry(model)

    mesh = meshes[0]
    renderable_obj = RenderableMesh(mesh)
    engine.clear_renderables()
    engine.add_renderable(renderable_obj)
    engine.start_rendering_loop()


config = load_config()

if config['RENDERING_ENGINE_ON']:
    engine = RenderEngine()
else:
    engine = None

trainer = Train(config=config, engine=engine)
trainer.train()
