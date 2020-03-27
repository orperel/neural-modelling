import numpy as np
from openmesh import *
import networkx as nx


def mesh_to_entry(mesh, mesh_name):
    points = np.stack([mesh.point(vh) for vh in mesh.vertices()])[:, :2]
    triangles = []
    for fh in mesh.faces():
        face_vids = [vh.idx() for vh in mesh.fv(fh)]  # TODO: Are they clockwise..?
        face_mesh_vids = [vid for vid in face_vids]
        triangles.append(face_mesh_vids)
    triangles = np.array(triangles)

    entry = dict(object_name=mesh_name,
                 points=points,
                 triangles=triangles)
    return entry


def entry_to_mesh(raw_entry):
    points_3d = np.concatenate((raw_entry['points'], np.zeros((raw_entry['points'].shape[0], 1))), axis=1)
    mesh = TriMesh(points=points_3d, face_vertex_indices=raw_entry['triangles'])
    return mesh


def construct_basic_mesh(mesh):
    basic_mesh = TriMesh()
    vid_mapping = dict()

    # Iterate over all faces
    for fh in mesh.faces():
        mesh_face_vids = [vh.idx() for vh in mesh.fv(fh)]
        # Add new vertices to mesh..
        for v_id in mesh_face_vids:
            if not v_id in vid_mapping:
                v_handle = mesh.vertex_handle(v_id)
                v_pos = mesh.point(v_handle)
                basic_v_handle = basic_mesh.add_vertex(v_pos)
                vid_mapping[v_id] = basic_v_handle.idx()
        basic_face_vids = tuple([basic_mesh.vertex_handle(vid_mapping[v_id]) for v_id in mesh_face_vids])
        if len(basic_face_vids) >= 3:  # Only valid faces..
            basic_mesh.add_face(basic_face_vids)

    return basic_mesh, vid_mapping


def _new_vertex_positions(v, tx, ty, tz):
    v0 = v + np.array((tx, ty, tz))
    v1 = v
    return v0, v1


def execute_vertex_split(mesh, v1_idx, vl_idx, vr_idx, tx, ty, tz):
    v1 = mesh.vertex_handle(v1_idx)
    vl = mesh.vertex_handle(vl_idx) if vl_idx != -1 else mesh.InvalidVertexHandle
    vr = mesh.vertex_handle(vr_idx) if vr_idx != -1 else mesh.InvalidVertexHandle
    v1_pos = mesh.point(v1)
    v0_pos, v1_pos = _new_vertex_positions(v1_pos, tx, ty, tz)
    halfedge_handle = mesh.vertex_split(v0_pos, v1, vl, vr)
    return halfedge_handle


def decimation_list_to_modifiers(self, mesh, decimation_list):
    basic_mesh, _ = self.construct_basic_mesh(mesh)
    seq_mesh, vid_mapping = self.construct_basic_mesh(mesh)
    modifiers = []

    for decimation_step in reversed(decimation_list):
        # middle = (openmesh_trimesh.point(decimation_step.v1) - openmesh_trimesh.point(decimation_step.v0)) / 2
        translation = mesh.point(decimation_step.v0) - mesh.point(decimation_step.v1)

        v1_idx = vid_mapping.get(decimation_step.v1.idx(), -1)
        vl_idx = vid_mapping.get(decimation_step.vl.idx(), -1)
        vr_idx = vid_mapping.get(decimation_step.vr.idx(), -1)
        split_vertex_modifier = dict(v1_idx=v1_idx,
                                     vl_idx=vl_idx,
                                     vr_idx=vr_idx,
                                     tx=translation[0], ty=translation[1], tz=translation[2])
        he_handle = self.execute_vertex_split(seq_mesh, v1_idx, vl_idx, vr_idx,
                                              translation[0], translation[1], translation[2])
        vid_mapping[decimation_step.v0.idx()] = seq_mesh.from_vertex_handle(he_handle).idx()
        vid_mapping[decimation_step.v1.idx()] = seq_mesh.to_vertex_handle(he_handle).idx()
        modifiers.append(split_vertex_modifier)

    return basic_mesh, modifiers


def number_of_connected_components(mesh):
    adjacency = {v.idx(): [u.idx() for u in mesh.vv(v)] for v in mesh.vertices()}
    edges = [(u, v) for u, children in adjacency.items() for v in children]
    G = nx.Graph()
    G.add_edges_from(edges)

    return nx.number_connected_components(G)


def simplify(self, raw_entry):
    mesh = entry_to_mesh(raw_entry)

    decimator = TriMeshDecimater(mesh)
    hModQuadric = TriMeshModQuadricHandle()
    hModProgressive = TriMeshModProgMeshHandle()
    decimator.add(hModQuadric)
    decimator.add(hModProgressive)

    decimator.module(hModQuadric).unset_max_err()
    decimator.initialize()

    decimator.decimate(n_collapses=mesh.n_vertices() - 4)
    decimation_list = decimator.module(hModProgressive).infolist()

    basic_mesh, modifiers = self.decimation_list_to_modifiers(mesh, decimation_list)
    mesh.garbage_collection()
    return basic_mesh, modifiers
