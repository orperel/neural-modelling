from graphics.render.interactive import InteractiveWidget
from graphics.render.render_engine import RenderEngine
from graphics.units.procedural_primitives import ProceduralPrimitives
from src.data.noisy_primitives_dataset import NoisyPrimitivesDataset

engine = RenderEngine()
dataset = NoisyPrimitivesDataset()
cube = dataset.generate_noisy_cube()
interactive = InteractiveWidget(engine)
# model = engine.load_model(path="/data/egg/lego.egg")
# parse_model_geometry(model)
engine.add_mesh(cube)
engine.start_rendering_loop()
