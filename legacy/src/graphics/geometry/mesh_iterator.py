
def iterate_faces_of_vertex_group(mesh, v_ids):
    iterated_face_ids = set()

    for v_id in v_ids:
        e_ids = mesh.vertices_to_edges[v_id]

        for e_id in e_ids:
            f_ids = mesh.edges_to_faces[e_id]
            for f_id in f_ids:
                if f_id in iterated_face_ids:
                    continue
                else:
                    iterated_face_ids.add(f_id)
                    yield f_id


def get_face_vertex_ids(mesh, f_id):

    v_ids = set()

    face = mesh.faces[f_id]
    for e_id in face:
        e = mesh.edges[e_id]
        vertex_a = e[0]
        vertex_b = e[1]
        v_ids.add(vertex_a)
        v_ids.add(vertex_b)
    return v_ids


def get_vertex_neighbors(mesh, v_id):

    neighbors = set()

    edges = mesh.vertices_to_edges[v_id]
    for e_id in edges:
        edge = mesh.edges[e_id]
        if edge[0] == v_id:
            neighbors.add(edge[1])
        else:
            neighbors.add(edge[0])
    return neighbors


def iterate_mesh_vertices(mesh):
    for v_id, vertex in enumerate(mesh.vertices):
        yield v_id, vertex


def get_edge_from_vertex_ids(mesh, v_id_1, v_id_2):
    """ Get the edge common to both vertices. Returns None if there is no common edge between the vertices """
    v1_edges = set(mesh.vertices_to_edges[v_id_1])
    v2_edges = set(mesh.vertices_to_edges[v_id_2])
    common_edges = tuple(v1_edges.intersection(v2_edges))

    if len(common_edges) == 0:
        return None
    else:
        (edge,) = common_edges
        return edge


def get_face_from_edge_ids(mesh, e_id_1, e_id_2, e_id_3):
    """ Get the face common to all edges. Returns None if there is no common face between the edges """
    e1_faces = set(mesh.edges_to_faces[e_id_1])
    e2_faces = set(mesh.edges_to_faces[e_id_2])
    e3_faces = set(mesh.edges_to_faces[e_id_3])

    common_faces = tuple(e1_faces.intersection(e2_faces).intersection(e3_faces))

    if len(common_faces) == 0:
        return None
    else:
        (face,) = common_faces
        return face


def get_face_from_vertex_ids(mesh, v_id_1, v_id_2, v_id_3):
    """ Get the face common to all vertices. Returns None if there is no common face between the vertices """
    e12 = get_edge_from_vertex_ids(mesh, v_id_1, v_id_2)
    e13 = get_edge_from_vertex_ids(mesh, v_id_1, v_id_3)
    e23 = get_edge_from_vertex_ids(mesh, v_id_2, v_id_3)

    if e12 is None or e13 is None or e23 is None:
        return None

    return get_face_from_edge_ids(e12, e13, e23)
