import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from celluloid import Camera
from src.data.visvalingam.shapes_simplification import get_points_in_order


def draw_2d_polyline(entry, ax, with_vertices=True):

    points = entry['points']
    verts = [points[0], *points, points[-1]]
    codes = [Path.MOVETO, *([Path.LINETO] * len(points)), Path.CLOSEPOLY]

    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='orange', edgecolor='k', lw=2)
    ax.add_patch(patch)

    if with_vertices:
        xs, ys = zip(*verts)
        ax.plot(xs, ys, marker='o', markerfacecolor='w', markeredgecolor='black')
        patch.set_edgecolor('black')

    title = entry["object_name"]
    ax.text(0.25, 1.025, f'{title}', transform=ax.transAxes, fontsize=14)
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)
    # plt.show()


def get_points_in_order_no_sort(constructed_order, limit=None):
    if limit:
        return constructed_order[:limit]
    else:
        return constructed_order


def render_polyline_animation(entry, constructed_order, output_dir):
    fig, ax = plt.subplots()
    camera = Camera(fig)
    meshname = entry['object_name']
    point_indices = get_points_in_order(constructed_order, limit=3)
    progressive = dict(object_name=entry['object_name'], points=entry['points'][point_indices])
    draw_2d_polyline(progressive, ax)
    camera.snap()

    for cut_idx in range(4, len(constructed_order)):
        point_indices = get_points_in_order(constructed_order, limit=cut_idx)
        progressive = dict(object_name=f'{meshname} - After Modifier #{cut_idx - 3}',
                           points=entry['points'][point_indices])
        draw_2d_polyline(progressive, ax)
        camera.snap()

    progressive = dict(object_name=f'{meshname} - Output', points=entry['points'][point_indices])
    draw_2d_polyline(progressive, ax, with_vertices=False)
    camera.snap()
    animation = camera.animate()
    outdir = os.path.join(output_dir, 'animations')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'{meshname}.gif')
    animation.save(outpath, writer='imagemagick')


def render_reconstruction_animation(mesh, modifiers, output_dir):
    mesh = mesh.copy()
    fig, ax = plt.subplots()
    camera = Camera(fig)
    meshname = mesh['object_name']
    vertices_order = mesh['polyline'].tolist()
    # point_indices = get_points_in_order_no_sort(constructed_order, limit=3)
    progressive = dict(object_name=mesh['object_name'], points=mesh['points'][vertices_order])
    draw_2d_polyline(progressive, ax)
    camera.snap()

    for frame_idx, modifier in enumerate(modifiers):
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

        progressive = dict(object_name=f'{meshname} - After Modifier #{frame_idx + 1}',
                           points=mesh['points'][vertices_order])
        draw_2d_polyline(progressive, ax)
        camera.snap()

    mesh['polyline'] = np.array(vertices_order)

    progressive = dict(object_name=f'{meshname} - Output', points=mesh['points'][vertices_order])
    draw_2d_polyline(progressive, ax, with_vertices=False)
    camera.snap()
    animation = camera.animate()
    outdir = os.path.join(output_dir, 'animations')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'{meshname}_reconstruction.gif')
    animation.save(outpath, writer='imagemagick')
