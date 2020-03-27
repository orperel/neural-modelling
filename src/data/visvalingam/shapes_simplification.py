import numpy as np
from src.data.visvalingam.visvalingam import Simplifier


def get_points_in_order(constructed_order, limit=None):
    if limit:
        return np.sort(constructed_order[:limit])
    else:
        return np.sort(constructed_order)


def generate_modifiers(raw_entry, constructed_order):
    points = raw_entry['points']

    # Map each vertex in original data to a new index, determined bt split order
    reconstruction_indices_mapping = {original_idx: reconstruction_idx
                                      for reconstruction_idx, original_idx in enumerate(constructed_order)}
    modifiers = []

    # 4th index is split out of 3rd first
    for cut_idx in range(4, len(constructed_order)):
        # All existing vertices so far, including the new split vertex
        current_points_order = get_points_in_order(constructed_order, limit=cut_idx)
        v0_orig_idx = constructed_order[cut_idx - 1]   # Index of the newly split vertex (original idx)

        # Location of newly split index within current_points_order, as well as adjacent vertices before and after it
        v0_orig_sortedloc = np.searchsorted(current_points_order, v0_orig_idx)
        adj1_orig_idx = current_points_order[v0_orig_sortedloc - 1]
        adj2_orig_idx = current_points_order[(v0_orig_sortedloc + 1) % len(current_points_order)]

        # Check which point is closer to v0 (of the 2 adjacent vertices) and set it as v1
        v0_pt = points[v0_orig_idx]
        adj1_pt = points[adj1_orig_idx]
        adj2_pt = points[adj2_orig_idx]
        adj1_to_v1_dist = np.linalg.norm(adj1_pt - v0_pt)
        adj2_to_v1_dist = np.linalg.norm(adj2_pt - v0_pt)
        if adj1_to_v1_dist < adj2_to_v1_dist:
            v1_pt = adj1_pt
            v1_idx = reconstruction_indices_mapping[adj1_orig_idx]
        else:
            v1_pt = adj2_pt
            v1_idx = reconstruction_indices_mapping[adj2_orig_idx]

        vl_idx = reconstruction_indices_mapping[adj1_orig_idx]
        vr_idx = reconstruction_indices_mapping[adj2_orig_idx]

        # Always split from v1 by convention, vl will connect to new vert
        tx = v1_pt[0] - v0_pt[0]
        ty = v1_pt[1] - v0_pt[1]

        modifier = dict(v1_idx=v1_idx,
                        vl_idx=vl_idx,
                        vr_idx=vr_idx,
                        tx=tx, ty=ty, tz=0)

        modifiers.append(modifier)

    current_points_order = get_points_in_order(constructed_order, limit=3)
    polyline_order = np.searchsorted(current_points_order, constructed_order[:3])
    basic_mesh = dict(object_name=raw_entry['object_name'],
                      points=raw_entry['points'][constructed_order[:3]],
                      polyline=polyline_order)

    return basic_mesh, modifiers


def rebuild_mesh(basic_mesh, modifiers):

    mesh = basic_mesh
    vertices_order = mesh['polyline'].tolist()

    for modifier in modifiers:
        v1_idx = modifier['v1_idx']
        vl_idx = modifier['vl_idx']
        vr_idx = modifier['vr_idx']
        tx = modifier['tx']
        ty = modifier['ty']

        v1 = mesh['points'][v1_idx]
        v0 = v1 - np.array((tx, ty))
        v0_idx = len(mesh['points'])

        # Slowdown alert: copy occurs every iteration here..
        mesh['points'] = np.append(mesh['points'], [v0], axis=0)

        insert_loc = vertices_order.index(vl_idx) + 1
        vertices_order.insert(insert_loc, v0_idx)

    mesh['polyline'] = np.array(vertices_order)
    return mesh


def simplify_mesh(raw_entry, output_path):
    points = raw_entry['points']

    # Avoid duplicates at start / end
    if (points[0] == points[-1]).all():
        points = points[:-1]

    simplifier = Simplifier(points)
    constructed_order = np.argsort(simplifier.thresholds)[::-1]
    # render_polyline_animation(raw_entry, constructed_order, output_path)
    basic_mesh, modifiers = generate_modifiers(raw_entry, constructed_order)
    # render_reconstruction_animation(basic_mesh, modifiers, output_path)
    # mesh = rebuild_mesh(basic_mesh, modifiers)
    return basic_mesh, modifiers
