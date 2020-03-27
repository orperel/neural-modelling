from src.graphics import Mesh
from src.graphics.modifiers.abstract_modifier import AbstractModifier


class ContractVertexPairModifier(AbstractModifier):

    def __init__(self, mesh: Mesh, v1_id, v2_id):
        super().__init__(mesh)
        self.v1_id = v1_id
        self.v2_id = v2_id
        self.v1 = self.mesh.vertices[self.v1_id]
        self.v2 = self.mesh.vertices[self.v2_id]
        self.v_id = None

        self.tx = None
        self.ty = None
        self.tz = None

    @staticmethod
    def _avg_vertex(v1, v2):
        v_new = (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])
        v_new = tuple(v_dim * 0.5 for v_dim in v_new)
        return v_new

    def merge_v2_into_v1(self):
        """ Associate v2's edges and faces with v1 """
        v2_edges = list(self.mesh.vertices_to_edges[self.v2_id])
        for e_id in v2_edges:
            e = self.mesh.edges[e_id]

            if (e[0] == self.v1_id and e[1] == self.v2_id) or (e[0] == self.v2_id and e[1] == self.v1_id):
                continue

            if e[0] == self.v2_id:
                self.mesh.update_edge(e_id, 0, self.v1_id)
            else:
                self.mesh.update_edge(e_id, 1, self.v1_id)

        # Remove v2 and edge connecting v1 and v2
        self.mesh.remove_vertex(self.v2_id)

    def remove_duplicate_edges(self):
        """ Remove duplicate edges and faces due to union of v1 and v2 """
        v1_neighbors = set()
        for e_id in self.mesh.vertices_to_edges[self.v1_id]:
            e = self.mesh.edges[e_id]

            # Other vertex id connected to v
            vi_id = e[1] if e[0] == self.v1_id else e[0]

            # If it was already encountered this is a double edge - remove it
            if vi_id in v1_neighbors:
                self.mesh.remove_edge(e_id)
            else:
                v1_neighbors.add(vi_id)

    def execute(self):

        v1 = self.v1
        v2 = self.v2

        # Calculate new vertex position
        v = self._avg_vertex(v1, v2)

        # Update v1 as v
        self.mesh.vertices[self.v1_id] = v
        self.merge_v2_into_v1()
        self.remove_duplicate_edges()

        # Store contract-translation params and other params for undo & visualizations
        tx, ty, tz = v[0] - v1[0], v[1] - v1[1], v[2] - v1[2]
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.v_id = self.v1_id

        return self.mesh

    def undo(self):
        from src.graphics.modifiers import SplitVertexModifier
        return SplitVertexModifier(mesh=self.mesh, v_id=self.v1_id, tx=self.tx, ty=self.ty, tz=self.tz)
