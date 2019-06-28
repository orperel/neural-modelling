import copy
from graphics.geometry.mesh import Mesh
from graphics.geometry.mesh_utils import vertex_distance
from graphics.geometry.mesh_iterator import iterate_faces_of_vertex_group, get_face_vertex_ids, get_vertex_neighbors
from graphics.modifiers.abstract_modifier import AbstractModifier


class SplitVertexModifier(AbstractModifier):

    def __init__(self, mesh: Mesh, v_id, tx, ty, tz):
        super().__init__(mesh)
        self.v_id = v_id
        self.v_data = self.mesh.vertices[self.v_id]
        self.tx = tx
        self.ty = ty
        self.tz = tz

        self.new_v1_id = None
        self.new_v2_id = None

    @staticmethod
    def _new_vertex_positions(v, tx, ty, tz):
        v1 = (v[0] + tx, v[1] + ty, v[2] + tz)
        v2 = (v[0] - tx, v[1] - ty, v[2] - tz)
        return v1, v2

    def _update_edges(self, v_id, v1, v2):
        v_edges = self.mesh.vertices_to_edges[v_id]  # All edges of v
        adj_verts = []                               # Adjacent vertices ids to the original v (connected by edge)

        v1_id = v_id                        # v1 will replace the original v
        v2_id = self.mesh.add_vertex(v2)    # Add v2 to the mesh

        # Update each edge to point to v1 or v2, and keep its vertex in adj_verts
        for e_id in v_edges:
            e = self.mesh.edges[e_id]
            vi_id = e[0] if e[0] != v_id else e[1]   # Find vi connecting to v
            adj_verts.append(vi_id)
            vi = self.mesh.vertices[vi_id]

            # Detach e(v, vi) and reconnect to v1 or v2 (the closer one)
            vj_id = v1_id if vertex_distance(v1, vi) <= vertex_distance(v2, vi) else v2_id

            # TODO: Handle function update here
            self.mesh.update_edge(e_id, vi_id, vj_id)  # Note: faces must be fixed after this function call!

        # Set v1 as the original v
        self.mesh.vertices[v1_id] = v1

        # Add a new edge connecting both
        v1_v2_edge_id = self.mesh.add_edge(e=(v1_id, v2_id))

        return v1_id, v2_id, adj_verts, v1_v2_edge_id

    def triangulate_quad_face(self, f_id, v1_v4_eid, v1, v2, v3, v4):
        """
        Triangulates a quad face into 2 new triangular faces.
        An edge will be formed between v1 and v3.
        The faces: (v1, v2, v3) and (v1, v3, v4) will be added instead of face(f_id).

            v2              v3
            /--------------.
           /  f1       .  /
          /       .      /
         /   .     f2   /
        /.-------------/
       v1              v4

        :param f_id: Quad face id
        :param v1_v4_eid: Edge id of edge connecting v1 and v4
        :param v1: vertex_id
        :param v2: vertex_id
        :param v3: vertex_id
        :param v4: vertex_id
        """

        e_new = (v1, v3)
        e_new_id = self.mesh.add_edge(e_new)

        # Update face #1
        coord_to_update = None
        # Find edge to replace with e_new
        for e_coord_idx, e_id in enumerate(self.mesh.faces[f_id]):
            e = self.mesh.edges[e_id]
            if v4 in e and v3 in e:
                replaced_eid = e_id
                coord_to_update = e_coord_idx
                break

        face1_edges = copy.deepcopy(self.mesh.faces[f_id])
        face1_edges[coord_to_update] = e_new_id
        self.mesh.update_face_from_edges(f_id, *face1_edges)

        # Add face #2
        self.mesh.add_face_from_edges(e1_id=v1_v4_eid,
                                      e2_id=replaced_eid,
                                      e3_id=e_new_id)

    def _find_invalid_faces(self, vertex_group):
        """ Finds all invalid faces associated with this vertex group """
        face_to_rectify = set()

        # Go over all faces associated with the vertices
        for f_id in iterate_faces_of_vertex_group(self.mesh, vertex_group):
            face_vertex_group = get_face_vertex_ids(self.mesh, f_id)

            # Assume only triangular faces are supported in the mesh.
            # If we have a face with more vertices here, it's necessarily invalid due to the split
            if len(face_vertex_group) > 3:
                face_to_rectify.add(f_id)

        return face_to_rectify

    def _update_faces(self, v1_v2_edge_id, adj_verts):

        v1_id, v2_id = self.mesh.edges[v1_v2_edge_id]
        face_to_rectify = self._find_invalid_faces(adj_verts)

        # Each face here has 4 vertices, and 3 edges, so we disband and rebuild it as 2 faces
        for f_id in face_to_rectify:
            face_vertex_group = get_face_vertex_ids(self.mesh, f_id)   # Get v1, v2, vi, vj

            # Calculate which vertex is the opposite of v1, and which is the opposite of v2
            # The following idiom: (var, ) = .. makes sure the result contains exactly one vertex id
            (v1_opposite_vid,) = face_vertex_group - get_vertex_neighbors(self.mesh, v1_id) - set([v1_id])
            (v2_opposite_vid,) = face_vertex_group - get_vertex_neighbors(self.mesh, v2_id) - set([v2_id])
            v1 = self.mesh.vertices[v1_id]
            v1_opposite = self.mesh.vertices[v1_opposite_vid]
            v2 = self.mesh.vertices[v2_id]
            v2_opposite = self.mesh.vertices[v2_opposite_vid]
            if vertex_distance(v1, v1_opposite) <= vertex_distance(v2, v2_opposite):
                self.triangulate_quad_face(f_id=f_id,
                                           v1_v4_eid=v1_v2_edge_id,
                                           v1=v1_id,
                                           v2=v2_opposite_vid,
                                           v3=v1_opposite_vid,
                                           v4=v2_id)

            else:
                self.triangulate_quad_face(f_id=f_id,
                                           v1_v4_eid=v1_v2_edge_id,
                                           v1=v2_id,
                                           v2=v1_opposite_vid,
                                           v3=v2_opposite_vid,
                                           v4=v1_id)

    def execute(self):
        v = self.v_data
        v1, v2 = self._new_vertex_positions(v, self.tx, self.ty, self.tz)

        v1_id, v2_id, adj_verts, v1_v2_edge_id = self._update_edges(self.v_id, v1, v2)
        self._update_faces(v1_v2_edge_id, adj_verts)

        self.new_v1_id = v1_id
        self.new_v2_id = v2_id

        return self.mesh

    def undo(self):
        from graphics.modifiers.contract_vertex_pair import ContractVertexPairModifier
        return ContractVertexPairModifier(mesh=self.mesh(), v1_id=self.new_v1_id, v2_id=self.new_v2_id)
