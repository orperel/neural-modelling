from graphics.geometry.mesh import Mesh
from graphics.modifiers.abstract_modifier import AbstractModifier


class SplitEdgeModifier(AbstractModifier):

    def __init__(self, mesh: Mesh, e_id):
        super().__init__(mesh)
        self.e_id = e_id
        self.edge = self.get_edge_data(e_id)

        # Will hold information of removed edge, to enable restoring it later, as well as the newly added vertex
        self.associated_faces, self.edges_of_associated_faces = None, None
        self.new_vid, self.new_edges = None, []

    @staticmethod
    def _avg_vertex(v1, v2):
        v_new = (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])
        v_new = tuple(v_dim * 0.5 for v_dim in v_new)
        return v_new

    def get_edge_data(self, e_id):
        edge = self.mesh.edges[e_id]
        v0_id, v1_id = edge
        v0_coords = self.mesh.vertices[v0_id]
        v1_coords = self.mesh.vertices[v1_id]
        return v0_coords, v1_coords

    def execute(self):

        edge_to_split = self.mesh.edges[self.e_id]
        v0, v1 = edge_to_split
        v0_coords, v1_coords = self.get_edge_data(self.e_id)

        v_new = self._avg_vertex(v0_coords, v1_coords)

        # First remove the edge
        edge_to_split, associated_faces, edges_of_associated_faces = self.mesh.remove_edge(self.e_id)

        # Then register the new vertex, and add 2 new edges
        v_new_id = self.mesh.add_vertex(v=v_new)
        new_edge_1 = (v0, v_new_id)
        new_edge_2 = (v0, v_new_id)
        new_eid_1 = self.mesh.add_edge(e=new_edge_1)
        new_eid_2 = self.mesh.add_edge(e=new_edge_2)
        self.new_edges.append(new_eid_1)
        self.new_edges.append(new_eid_2)

        # Next add 2 new faces in stead of each original face
        for face in associated_faces.values():

            third_vertex = None
            for v in face:
                if v != v0 and v != v1:
                    third_vertex = v
                    break

            #   v0   ----- v_new_id--- v1
            #       \        |        /
            #         \      |      /
            #           \    |    /
            #             \  |  /
            #               \|/
            #            third_vertex
            self.mesh.add_face(f=(v0, v_new_id, third_vertex))
            self.mesh.add_face(f=(v_new_id, v1, third_vertex))

        self.new_vid = v_new_id

        self.edge = [v0_coords, v1_coords]
        self.associated_faces = associated_faces
        self.edges_of_associated_faces = edges_of_associated_faces

        return self.mesh

    def undo(self):

        # Remove the added vertex
        self.mesh.remove_vertex(self.new_vid)
        self.mesh.restore_edge(self.e_id, self.edge, self.associated_faces, self.edges_of_associated_faces)

        return self.mesh
