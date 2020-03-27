import os
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from celluloid import Camera
import copy
from src.data.openmesh.shapes_simplification import execute_vertex_split, mesh_to_entry


def draw_2d_mesh(entry, ax):
    title = entry["object_name"]
    triang = mtri.Triangulation(x=entry['points'][:, 0], y=entry['points'][:, 1], triangles=entry['triangles'])
    ax.triplot(triang, 'ko-')
    # ax.set_title(f'{title}')
    ax.text(0.25, 1.025, f'{title}', transform=ax.transAxes, fontsize=14)


def render_entry(entry, output_dir):
    entry_name = entry["object_name"]
    outdir = os.path.join(output_dir, 'renders')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(output_dir, 'renders', entry_name)
    fig, ax = plt.subplots()
    draw_2d_mesh(entry, ax)
    plt.savefig(f'{outpath}')


def render_mesh_animation(mesh, modifiers, output_dir, meshname):
    mesh = copy.deepcopy(mesh)  # Retain the original
    fig, ax = plt.subplots()
    camera = Camera(fig)
    mesh_entry = mesh_to_entry(mesh, f'{meshname} - Start')
    draw_2d_mesh(mesh_entry, ax)
    camera.snap()

    for m_count, m in enumerate(modifiers):
        execute_vertex_split(mesh, m['v1_idx'], m['vl_idx'], m['vr_idx'], m['tx'], m['ty'], m['tz'])
        mesh_entry = mesh_to_entry(mesh, f'{meshname} - After Modifier #{m_count + 1}')
        draw_2d_mesh(mesh_entry, ax)
        camera.snap()
    animation = camera.animate()
    outdir = os.path.join(output_dir, 'animations')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'{meshname}.gif')
    animation.save(outpath, writer='imagemagick')

