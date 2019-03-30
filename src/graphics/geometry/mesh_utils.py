import numpy as np

eps = 1e-8
pi = np.pi
up_vector = np.array(0, 0, 1)


def edge(v1, v2):
    return (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])


def sort_vertices_in_clockwise_order(v1, v2, v3):
    """
    Sorts vertices in clockwise order.
    1) Calculate normal, by cross product of 2 edges formed by the 3 vertices.
    2) Normalize by calculating magnitude.
    3) Calculate dot product of normal and predefined up vector.

    Reference:
    https://stackoverflow.com/questions/1516296/find-the-normal-angle-of-the-face-of-a-triangle-in-3d-given-the-co-ordinates-of
    """
    edge1 = edge(v1, v2)
    edge2 = edge(v1, v3)
    normal = np.cross(edge1, edge2)

    magnitude = np.linalg.norm(normal)

    if magnitude < eps:
        raise ValueError('Degenerated triangle given to sort_vertices_in_clockwise_order, most probably 2 vertices are'
                         'very close to each other.')

    # Calculate unit vector of normal
    normal = normal / magnitude

    # Angle between unit normal and unit up vector in simply the arccosine of dot product
    slope = np.arccos(np.dot(normal, up_vector))

    if 0 <= slope <= pi:
        return v1, v2, v3   # CW
    else:
        return v3, v2, v1   # CCW, invert to get CW
