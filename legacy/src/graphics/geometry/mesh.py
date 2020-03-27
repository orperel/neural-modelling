import itertools


class Mesh:

    def __init__(self):
        self.vertices = list()
        self.edges = list()
        self.faces = list()

        self.vertices_to_edges = {}  # Vertex id to edge id
        self.edges_to_faces = {}     # Edge id to face id

    def add_vertex(self, v):
        """
        :param v: tuple of 3 float coordinates (x,y,z)
        :return: The assigned vertex id of the new vertex.
        """
        v_id = len(self.vertices)
        self.vertices.append(v)
        return v_id

    def get_edge_id(self, v1_id, v2_id):
        """
        Return edge id of edge connecting between the 2 vertices, if such as edge exists.
        Otherwise None is returned.
        This function operates in O(E) worst case, but for most practical cases, we assume each vertex has
        a very small number of edges connecting it.
        :param v1_id:
        :param v2_id:
        :return: Edge id or None
        """

        for v_id in (v1_id, v2_id):
            if v_id not in self.vertices_to_edges:
                continue    # Vertex is not connected to any edge
            connected_edge_ids = self.vertices_to_edges[v_id]
            for edge_id in connected_edge_ids:
                edge = self.edges[edge_id]
                if edge[0] == v1_id and edge[1] == v2_id or edge[0] == v2_id and edge[1] == v1_id:
                    return edge_id

        return None

    def add_edge(self, e):
        """
        Adds an edge between the 2 vertices, if it doesn't exist already.
        :param e: tuple of 2 vertex indices
        :return edge id of newly created edge
        """
        e_id = self.get_edge_id(*e)
        if e_id is None:
            e_id = len(self.edges)  # Register new edge id
            self.edges.append(e)
            for vertex in e:
                self.vertices_to_edges.setdefault(vertex, []).append(e_id)

        return e_id

    def update_edge(self, e_id, edge_coord, new_v_id):
        """
        Updates an edge id to point at a new vertex at edge[edge_coord].
        :param e_id: Edge id of edge to update
        :param edge_coord: 0 or 1 (first vertex or second vertex is updated)
        :param new_v_id: v_id of new vertex the edge points at
        """
        old_e = self.edges[e_id]
        old_v_id = old_e[edge_coord]

        # Update vertices mappings
        self.vertices_to_edges[old_v_id].remove(e_id)
        self.vertices_to_edges.setdefault(new_v_id, []).append(e_id)

        # Update edge (replace old vertex with new)
        new_e = list(old_e)
        new_e[edge_coord] = new_v_id
        new_e = tuple(new_e)
        self.edges[e_id] = new_e

        # Update faces (replace old vertex with new)
        for f_id in self.edges_to_faces[e_id]:
            f = self.faces[f_id]
            f = [v_id if v_id != old_v_id else new_v_id for v_id in f]
            self.faces[f_id] = tuple(f)

    def add_face(self, f):
        """
        :param f: tuple of n vertex indices
        :return face id of newly created face
        """
        f_id = len(self.faces)
        self.faces.append(f)

        vertices_iter = iter(f)  # Iterate over pairs of vertices
        v_first = next(vertices_iter)
        v0 = v_first
        for v1 in vertices_iter:
            edge = (v0, v1)
            edge_id = self.add_edge(edge)
            self.edges_to_faces.setdefault(edge_id, []).append(f_id)
            v0 = v1

        v_last = v0
        edge = (v_last, v_first)
        edge_id = self.add_edge(edge)
        self.edges_to_faces.setdefault(edge_id, []).append(f_id)

        return f_id

    def _fetch_vertices_from_edge_group(self,  e1_id, e2_id, e3_id):
        # TODO: Make sure this is CW
        vertices = set()
        for e_id in (e1_id, e2_id, e3_id):
            v1, v2 = self.edges[e_id]
            vertices.add(v1)
            vertices.add(v2)
        return tuple(vertices)

    def add_face_from_edges(self, e1_id, e2_id, e3_id):
        """
        :param e1_id: e1 of the face to point to
        :param e2_id: e2 of the face to point to
        :param e3_id: e3 of the face to point to
        :return face id of newly created face
        """
        f = self._fetch_vertices_from_edge_group(e1_id, e2_id, e3_id)
        assert len(f) == 3

        f_id = len(self.faces)
        self.faces.append(f)

        self.edges_to_faces.setdefault(e1_id, []).append(f_id)
        self.edges_to_faces.setdefault(e2_id, []).append(f_id)
        self.edges_to_faces.setdefault(e3_id, []).append(f_id)

        return f_id

    def update_face_from_edges(self, f_id, e1_id, e2_id, e3_id):
        """
        Updates a face id to point at 3 new edges.
        :param f_id: Face id of edge to update
        :param e1: e1 of the face to point to
        :param e2: e2 of the face to point to
        :param e3: e3 of the face to point to
        """
        new_f = self._fetch_vertices_from_edge_group(e1_id, e2_id, e3_id)
        assert len(new_f) == 3

        face_edges = self._find_all_edge_ids_associated_with_face(f_id)
        for edge_id in face_edges:
            self.edges_to_faces[edge_id].remove(f_id)
        self.faces[f_id] = new_f
        for edge_id in (e1_id, e2_id, e3_id):
            self.edges_to_faces.setdefault(edge_id, []).append(f_id)

    def remove_vertex(self, v_id):
        """
        Removes the given vertex id from the vertex array, as well as all of it's associated edges and faces.
        :param v_id: Id of vertex to cancel
        """
        associated_edges_and_faces = []

        v_edges = list(self.vertices_to_edges[v_id])
        for e_id in v_edges:
            edge, associated_faces, edges_of_associated_faces = self.remove_edge(e_id)
            associated_edges_and_faces.append((edge, associated_faces, edges_of_associated_faces))

        self.vertices[v_id] = None
        return associated_edges_and_faces

    def _find_all_edge_ids_associated_with_face(self, f_id):
        associated_edge_ids = set()

        face_vertices_ids = self.faces[f_id]
        for v_id in face_vertices_ids:
            for e_id in self.vertices_to_edges[v_id]:
                for adjacent_fid in self.edges_to_faces.get(e_id, []):
                    if adjacent_fid == f_id:
                        associated_edge_ids.add(e_id)

        return associated_edge_ids

    def remove_edge(self, e_id):
        """
        Removes edge and all it's associated faces.
        The vertices composing the edge are not removed.
        Essentially, the edge will be canceled, and can therefore be restored later.
        :param e_id: Edge id.
        :return: (edge, associated_faces), representing:
         1) tuple of vertices ids of the edge
         2) mapping of face_id -> face information of associated faces
         3) mapping of face_id -> edge ids pointing at associated face

            #          e1        e2
            #     v1 ------ v2------- v3
            #       \        |        /
            #         \  f1  |  f2  /
            #        e3 \    |e5  / e4
            #             \  |  /
            #               \|/
            #               v4
            # i.e: Assume e5=(v2,v4) is removed. The return value is:
            #   - edge = (2,4)
            #   - associated_faces = { f1: {v1,v2,v4}, f2: {v2,v3,v4} }
            #   - edges_of_associated_faces = { f1: {e1,e3,e5}, f2: {e2,e4,e5} }
        """

        edge = self.edges[e_id]

        # Maintain some mapping of face_id -> face info for removed faces
        neighbour_faces = self.edges_to_faces[e_id]
        associated_faces = {f_id: self.faces[f_id] for f_id in neighbour_faces}
        edges_of_associated_faces = {f_id: self._find_all_edge_ids_associated_with_face(f_id)
                                     for f_id in neighbour_faces}

        # Remove face associations, all faces touching the edge will be removed as well
        e_faces = list(self.edges_to_faces[e_id])
        for f_id in e_faces:
            # Remove pointers from nearby edges pointing at the removed face
            # (note: these are the edges neighbouring the removed edge)
            for associated_e_id in edges_of_associated_faces[f_id]:
                self.edges_to_faces[associated_e_id].remove(f_id)
            self.faces[f_id] = None
        del self.edges_to_faces[e_id]

        # Remove vertex associations
        for v_id in edge:
            self.vertices_to_edges[v_id].remove(e_id)

        # Cancel edge
        self.edges[e_id] = None

        return edge, associated_faces, edges_of_associated_faces

    def restore_edge(self, e_id, edge, associated_faces, edges_of_associated_faces):
        """
        Restores an edge previously removed with remove_edge.
        All it's associated faces will be restored as well.
        :param e_id: The edge id to restore
        :param edge: The edge information: (v0, v1), for 2 existing vertex ids
        :param associated_faces:
        :param edges_of_associated_faces:
        """
        # Restore faces
        self.edges_to_faces[e_id] = []
        for f_id, face in associated_faces.items():
            self.faces[f_id] = face
            for neighbor_edge_id in edges_of_associated_faces[f_id]:
                self.edges_to_faces[neighbor_edge_id].append(f_id)

        # Restore vertex associations
        for v_id in edge:
            self.vertices_to_edges[v_id].append(e_id)

        # Restore edge
        self.edges[e_id] = edge
