from geometry import ProceduralPrimitives
from interactive import InteractiveWidget
from render_engine import RenderEngine

engine = RenderEngine()
ProceduralPrimitives.add_cube(engine)
interactive = InteractiveWidget(engine)
engine.start_rendering_loop()
