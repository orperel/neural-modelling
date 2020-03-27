import numpy as np
from openmesh import *
from src.graphics.modifiers.openmesh import OpenMeshSplitVertexModifier
from src.graphics import Mesh
from src.graphics.render.render_engine import RenderEngine
from src.graphics.render.interactive import InteractiveWidget
from src.graphics.render.animation import Animation


def openmesh_to_mesh(openmesh_trimesh):
    mesh = Mesh()
    vid_mapping = dict()
    for vh in openmesh_trimesh.vertices():
        openmesh_vid = vh.idx()
        openmesh_vpos = openmesh_trimesh.point(vh)
        mesh_id = mesh.add_vertex(tuple(openmesh_vpos))
        vid_mapping[openmesh_vid] = mesh_id

    # iterate over all faces
    for fh in openmesh_trimesh.faces():
        face_openmesh_vids = [vh.idx() for vh in openmesh_trimesh.fv(fh)]   # TODO: Are they clockwise..?
        face_mesh_vids = [vid_mapping[vid] for vid in face_openmesh_vids]
        if len(face_mesh_vids) >= 3:
            mesh.add_face(face_mesh_vids)

    return mesh, vid_mapping


def render_loop(modifiers):
    config = dict(RENDERING_ENGINE_ON=True, DISPLAY_ON=True)
    engine = RenderEngine(config)

    animation = Animation(engine, modifiers)
    interactive = InteractiveWidget(engine, animation)

    animation.start()


def normalize(mesh):
    model_v = np.stack([mesh.point(vh) for vh in mesh.vertices()])
    model_max = np.max(model_v, axis=0)
    model_min = np.min(model_v, axis=0)
    model_diameter = model_max - model_min
    model_center = np.mean(model_v, axis=0)

    max_dim = np.argmax(model_diameter)
    scale_factor = model_diameter[max_dim]

    for vh in mesh.vertices():
        normalized_pos = (mesh.point(vh) - model_center) / scale_factor
        mesh.set_point(vh, normalized_pos)


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
        if len(basic_face_vids) >= 3:   # Only valid faces..
            basic_mesh.add_face(basic_face_vids)

    return basic_mesh, vid_mapping

def decimation_list_to_modifiers(mesh, decimation_list):
    basic_mesh, _ = construct_basic_mesh(mesh)
    seq_mesh, vid_mapping = construct_basic_mesh(mesh)
    modifiers = []

    for decimation_step in reversed(decimation_list):
        # middle = (openmesh_trimesh.point(decimation_step.v1) - openmesh_trimesh.point(decimation_step.v0)) / 2
        translation = mesh.point(decimation_step.v0) - mesh.point(decimation_step.v1)

        v1_idx = vid_mapping[decimation_step.v1.idx()]
        vl_idx = vid_mapping[decimation_step.vl.idx()]
        vr_idx = vid_mapping[decimation_step.vr.idx()]
        split_vertex_modifier = OpenMeshSplitVertexModifier(mesh=seq_mesh,
                                                            v1_idx=v1_idx,
                                                            vl_idx=vl_idx,
                                                            vr_idx=vr_idx,
                                                            tx=translation[0], ty=translation[1], tz=translation[2])
        split_vertex_modifier.execute()
        he_handle = split_vertex_modifier.halfedge_handle
        vid_mapping[decimation_step.v0.idx()] = seq_mesh.from_vertex_handle(he_handle).idx()
        vid_mapping[decimation_step.v1.idx()] = seq_mesh.to_vertex_handle(he_handle).idx()
        split_vertex_modifier.mesh = basic_mesh
        split_vertex_modifier.halfedge_handle = None
        modifiers.append(split_vertex_modifier)

    return seq_mesh, modifiers


def foo():
    mesh = read_trimesh('../data/obj/bunny/bunny.obj')
    normalize(mesh)

    decimator = TriMeshDecimater(mesh)
    hModQuadric = TriMeshModQuadricHandle()
    hModProgressive = TriMeshModProgMeshHandle()
    decimator.add(hModQuadric)
    decimator.add(hModProgressive)

    decimator.module(hModQuadric).unset_max_err()
    decimator.initialize()

    decimator.decimate(n_collapses=mesh.n_vertices() - 8)
    decimation_list = decimator.module(hModProgressive).infolist()

    trainable_mesh, modifiers = decimation_list_to_modifiers(mesh, decimation_list)
    mesh.garbage_collection()
    render_loop(modifiers)

    # while len(list(mesh.vertices())) > 8:
    #     decimator.decimate(n_collapses=1)
    #     mesh.garbage_collection()
    #     print(len(list(mesh.vertices())))


foo()