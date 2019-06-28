from graphics.geometry.mesh import Mesh


class ProceduralPrimitives:

    def __init__(self):
        pass

    @staticmethod
    def generate_cube():
        """
        :return: A cube with the following structure of mesh (numbers representing vertex ids)

                                      6 ______________ 7
                                       /|            /|
                                      / |           / |
                                     /  |          /  |
            Y                     4 --------------- 5 |
            ^      Z                |   |         |   |
            |    ^                  | 2 |_________|___| 3
            |   /                   |  /          |  /
            |  /                    | /           | /
            | /                     |/            |/
            |/--------> X         0 --------------- 1

        """
        cube = Mesh()

        cube.add_vertex((-1, -1, -1))  # 0
        cube.add_vertex((1, -1, -1))  # 1
        cube.add_vertex((-1, 1, -1))  # 2
        cube.add_vertex((1, 1, -1))  # 3
        cube.add_vertex((-1, -1, 1))  # 4
        cube.add_vertex((1, -1, 1))  # 5
        cube.add_vertex((-1, 1, 1))  # 6
        cube.add_vertex((1, 1, 1))  # 7

        cube.add_face((4, 1, 0))
        cube.add_face((4, 5, 1))
        cube.add_face((6, 4, 0))
        cube.add_face((2, 6, 0))
        cube.add_face((5, 3, 1))
        cube.add_face((5, 7, 3))
        cube.add_face((7, 6, 3))
        cube.add_face((3, 6, 2))
        cube.add_face((6, 7, 5))
        cube.add_face((6, 5, 4))
        cube.add_face((2, 3, 1))
        cube.add_face((0, 2, 1))

        return cube
