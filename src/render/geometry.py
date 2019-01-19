class ProceduralPrimitives:

    def __init__(self):
        pass

    @staticmethod
    def add_cube(engine):
        engine.add_vertex(-1, -1, -1)  # 0
        engine.add_vertex(1, -1, -1)  # 1
        engine.add_vertex(-1, 1, -1)  # 2
        engine.add_vertex(1, 1, -1)  # 3
        engine.add_vertex(-1, -1, 1)  # 4
        engine.add_vertex(1, -1, 1)  # 5
        engine.add_vertex(-1, 1, 1)  # 6
        engine.add_vertex(1, 1, 1)  # 7

        engine.add_triangle(4, 1, 0)
        engine.add_triangle(4, 5, 1)
        engine.add_triangle(6, 4, 0)
        engine.add_triangle(2, 6, 0)
        engine.add_triangle(5, 3, 1)
        engine.add_triangle(5, 7, 3)
        engine.add_triangle(7, 6, 3)
        engine.add_triangle(3, 6, 2)
        engine.add_triangle(6, 7, 5)
        engine.add_triangle(6, 5, 4)
        engine.add_triangle(2, 3, 1)
        engine.add_triangle(0, 2, 1)
