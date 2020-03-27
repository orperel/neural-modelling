from src.graphics.render.render_engine import RenderEngine
from src.graphics import RenderableMesh
from src.graphics import parse_model_geometry


class Preprocess3D:

    def __init__(self):
        self.engine = RenderEngine()

    def decompose(self,  mesh):
        pass


def load_obj(engine, model_path="/data/egg/lego.egg"):
    model = engine.load_model(path=model_path)
    meshes = parse_model_geometry(model)

    mesh = meshes[0]

    simplifier = Preprocess3D()
    simplified_mesh = simplifier.decompose(mesh)

    renderable_obj = RenderableMesh(simplified_mesh)
    engine.clear_renderables()
    engine.add_renderable(renderable_obj)
    engine.start_rendering_loop()
