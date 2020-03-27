from src.graphics.render.abstract_renderable import AbstractRenderable


class OpenMeshRenderable(AbstractRenderable):

    def __init__(self, mesh=None, default_color=None, alpha=1.0):
        super().__init__(default_color, alpha)

        self.vid_mapping = dict()
        if mesh is not None:
            self.add_mesh(mesh)

    def add_mesh(self, mesh):
        for vh in mesh.vertices():
            openmesh_vid = vh.idx()
            openmesh_vpos = mesh.point(vh)
            self._add_vertex_data(*openmesh_vpos)
            self.vid_mapping[openmesh_vid] = len(self.vid_mapping)

        # iterate over all faces
        for fh in mesh.faces():
            face_openmesh_vids = [vh.idx() for vh in mesh.fv(fh)]

            # Avoid canceled faces
            if all([mesh.vertex_handle(vid).is_valid() for vid in face_openmesh_vids]):
                face_mesh_vids = [self.vid_mapping[vid] for vid in face_openmesh_vids]
                if len(face_mesh_vids) >= 3:
                    self._add_triangle(*face_mesh_vids)  # Assume faces to be triangular
