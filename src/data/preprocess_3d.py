from graphics.render.render_engine import RenderEngine
from graphics.units.mesh import Mesh


class Preprocess3D:

    def __init__(self):
        self.engine = RenderEngine()

    def handle_3d_model(self, model_egg_path):
        model_path = self.engine.load_model(path=model_egg_path)    # i.e: model_path = "/data/egg/lego.egg"

    def handle_3d_mesh(self, mesh: Mesh):

        ProceduralPrimitives.add_cube(engine)

    parse_model_geometry(model)
    engine.start_rendering_loop()
