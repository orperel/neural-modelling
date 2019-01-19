from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import *
from direct.interval.IntervalGlobal import *
from panda3d.core import lookAt
from panda3d.core import GeomVertexFormat, GeomVertexData
from panda3d.core import Geom, GeomTriangles, GeomVertexWriter
from panda3d.core import Texture, GeomNode
from panda3d.core import PerspectiveLens
from panda3d.core import CardMaker
from panda3d.core import Light, Spotlight
from panda3d.core import TextNode
from panda3d.core import LVector3
import sys
import os


class RenderEngine:

    def __init__(self):
        self.base = self.initialize_scene()

        format = GeomVertexFormat.getV3n3cp()
        self.vertex_data = GeomVertexData('model', format, Geom.UHDynamic)

        self.vertex_writer = GeomVertexWriter(self.vertex_data, 'vertex')
        self.normal_writer = GeomVertexWriter(self.vertex_data, 'normal')
        self.color_writer = GeomVertexWriter(self.vertex_data, 'color')

        self.triangle_list = GeomTriangles(Geom.UHDynamic)
        self.mesh = Geom(self.vertex_data)
        self.mesh.addPrimitive(self.triangle_list)

        self.mesh_node = GeomNode('model')
        self.mesh_node.addGeom(self.mesh)

        self.scene_graph_root = self.setup_rendering_loop()
        self.setup_lights()

    def start_rendering_loop(self):
        self.base.run()

    def setup_rendering_loop(self):
        scene_graph_root = render.attachNewNode(self.mesh_node)
        scene_graph_root.hprInterval(1.5, (360, 360, 360)).loop()

        # OpenGl by default only draws "front faces" (polygons whose vertices are
        # specified CCW).
        scene_graph_root.setTwoSided(True)
        return scene_graph_root

    def initialize_scene(self):
        base = ShowBase()
        base.disableMouse()
        base.camera.setPos(0, -10, 0)

        self.setup_hud(base)

        return base

    @staticmethod
    def setup_hud(base):
        title = OnscreenText(text="Neural Box Modeling GT demo",
                             style=1, fg=(1, 1, 1, 1), pos=(-0.1, 0.1), scale=.07,
                             parent=base.a2dBottomRight, align=TextNode.ARight)
        side_light_event = OnscreenText(text="1: Toggle Light from the front On/Off",
                                        style=1, fg=(1, 1, 1, 1), pos=(0.06, -0.08),
                                        align=TextNode.ALeft, scale=.05,
                                        parent=base.a2dTopLeft)
        top_light_event = OnscreenText(text="2: Toggle Light from on top On/Off",
                                       style=1, fg=(1, 1, 1, 1), pos=(0.06, -0.14),
                                       align=TextNode.ALeft, scale=.05,
                                       parent=base.a2dTopLeft)

    def setup_lights(self):
        slight = Spotlight('slight')
        slight.setColor((1, 1, 1, 1))
        lens = PerspectiveLens()
        slight.setLens(lens)
        self.light_side = render.attachNewNode(slight)
        self.light_top = render.attachNewNode(slight)
        self.is_light_side_on = False
        self.is_light_top_on = False
        self.toggleLightsSide()
        self.toggleLightsUp()

    @staticmethod
    def _normalized(*args):
        """ Helper function for normalizing inline """
        myVec = LVector3(*args)
        myVec.normalize()
        return myVec

    def add_vertex(self, x, y, z):
        self.vertex_writer.addData3(x, y, z)
        self.normal_writer.addData3(self._normalized(2 * x - 1, 2 * y - 1, 2 * z - 1))
        self.color_writer.addData4f(1.0, 0.0, 0.0, 1.0)

    def add_triangle(self, v1, v2, v3):
        """
        Adds a new triangle to the mesh
        :param v1: Index of vertex #1
        :param v2: Index of vertex #2
        :param v3: Index of vertex #3
        """
        self.triangle_list.addVertices(v1, v2, v3)

    def toggleLightsSide(self):
        self.is_light_side_on = not self.is_light_side_on

        if self.is_light_side_on:
            render.setLight(self.light_side)
            self.light_side.setPos(self.scene_graph_root, 10, -300, 0)
            self.light_side.lookAt(10, 0, 0)
        else:
            render.setLightOff(self.light_side)

    def toggleLightsUp(self):
        global cube
        self.is_light_top_on = not self.is_light_top_on

        if self.is_light_top_on:
            render.setLight(self.light_top)
            self.light_top.setPos(self.scene_graph_root, 10, 0, 300)
            self.light_top.lookAt(10, 0, 0)
        else:
            render.setLightOff(self.light_top)
