from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import *
from direct.interval.IntervalGlobal import *
from direct.task import Task
from panda3d.core import lookAt
from panda3d.core import Texture, GeomNode
from panda3d.core import PerspectiveLens, Camera
from panda3d.core import CardMaker
from panda3d.core import Light, Spotlight, AmbientLight
from panda3d.core import TextNode
from panda3d.core import Filename
from panda3d.core import MeshDrawer2D
from panda3d.core import ConfigVariableString
import sys
import os
import numpy as np
import time
from math import sin, cos, pi


class RenderEngine:
    def __init__(self, rendering_config):
        self.display_on = rendering_config['DISPLAY_ON']
        self._initialize_display(self.display_on)

        self.base = self.initialize_scene()

        self.renderables = []
        self.active_geom_node_paths = []
        self.scene_graph_root = None

        self.dr_top, self.dr_left, self.dr_front, self.dr_perspective = None, None, None, None
        self.hud = []

        self.wireframe_mode = 0

    @staticmethod
    def _initialize_display(display_on):
        if not display_on:
            ConfigVariableString("window-type", "offscreen").setValue("offscreen")
            ConfigVariableString("audio-library-name", "null").setValue("null")
            ConfigVariableString("load-display", "p3tinydisplay").setValue("p3tinydisplay")

    def add_renderable(self, renderable):
        self.renderables.append(renderable)

    def clear_renderables(self):
        for rend in self.renderables:
            rend.mesh_node.removeAllGeoms()
        for geom_node_path in self.active_geom_node_paths:
            geom_node_path.removeNode()
        self.active_geom_node_paths = []
        self.renderables = []
        self.scene_graph_root = None

    def prepare_for_rendering(self):
        self.setup_renderables()
        if self.dr_perspective is None:
            self.setup_lights()
            self.setup_splitscreen(invoke_perspective_rotation=True)
            self.setup_hud()

    def start_rendering_loop(self):
        self.prepare_for_rendering()
        self.base.run()

    def stop_rendering_loop(self):
        self.base.task_mgr.running = False

    def restart_rendering_loop(self):
        self.base.run()

    def run_for(self, seconds):
        self.prepare_for_rendering()
        start = time.time()

        while time.time() - start < seconds:
            self.base.task_mgr.step()

    def setup_renderables(self, loop_duration_secs=10):
        self.renderables[0].render()
        scene_graph_root = render.attachNewNode(self.renderables[0].mesh_node)
        self.active_geom_node_paths.append(scene_graph_root)
        for rend in self.renderables[1:]:
            rend.render()
            rend_geom_node = render.attachNewNode(rend.mesh_node)
            rend_geom_node.setTwoSided(True)
            rend_geom_node.setRenderModeThickness(5.0)
            self.active_geom_node_paths.append(rend_geom_node)

            alight = AmbientLight('alight')
            alight.setColor((1, 1, 1, 1))
            alight_node = rend_geom_node.attachNewNode(alight)
            rend_geom_node.setLight(alight_node)

        # OpenGl by default only draws "front faces" (polygons whose vertices are specified CCW).
        scene_graph_root.setTwoSided(True)

        self.scene_graph_root = scene_graph_root

    def initialize_scene(self):
        base = ShowBase()
        base.disableMouse()
        base.camera.setName("DefaultCamera")
        base.camera.setPos(0, -3, 0)

        return base

    def setup_splitscreen(self, invoke_perspective_rotation=False):
        base = self.base
        camera_pos_anchor = 3.0

        left_camera = Camera("Left")
        left_camera_node = render.attachNewNode(left_camera)
        left_camera_node.setName("LeftCamera")
        left_camera_node.setPos(camera_pos_anchor, 0, 0)
        left_camera_node.lookAt((0,0,0))

        top_camera = Camera("Top")
        top_camera_node = render.attachNewNode(top_camera)
        top_camera_node.setName("TopCamera")
        top_camera_node.setPos(0, 0, camera_pos_anchor)
        top_camera_node.lookAt((0,0,0))

        front_camera = Camera("Front")
        front_camera_node = render.attachNewNode(front_camera)
        front_camera_node.setName("FrontCamera")
        front_camera_node.setPos(0, camera_pos_anchor, 0)
        front_camera_node.lookAt((0,0,0))

        perspective_camera = Camera("Perspective")
        perspective_camera_node = render.attachNewNode(perspective_camera)
        perspective_camera_node.setName("PerspectiveCamera")
        perspective_camera_node.setPos(camera_pos_anchor, camera_pos_anchor, camera_pos_anchor)
        perspective_camera_node.lookAt((0, 0, 0))

        # Disable default display region
        dr = base.camNode.getDisplayRegion(0)
        dr.setActive(0)  # Or leave it (dr.setActive(1))

        window = dr.getWindow()
        dr1 = window.makeDisplayRegion(0.0, 0.5, 0.0, 0.5)
        dr1.setSort(dr.getSort())
        dr2 = window.makeDisplayRegion(0.0, 0.5, 0.5, 1.0)
        dr2.setSort(dr.getSort())
        dr3 = window.makeDisplayRegion(0.5, 1.0, 0.0, 0.5)
        dr3.setSort(dr.getSort())
        dr4 = window.makeDisplayRegion(0.5, 1.0, 0.5, 1.0)
        dr4.setSort(dr.getSort())

        dr1.setCamera(top_camera_node)
        dr2.setCamera(front_camera_node)
        dr3.setCamera(perspective_camera_node)
        dr4.setCamera(left_camera_node)

        self.dr_top = dr4
        self.dr_left = dr1
        self.dr_perspective = dr3
        self.dr_front = dr2

        def spinCameraTask(task):
            angleDegrees = task.time * 30.0
            angleRadians = angleDegrees * (pi / 180.0)
            perspective_camera_node.setPos(camera_pos_anchor * sin(angleRadians),
                                           -camera_pos_anchor * cos(angleRadians),
                                           camera_pos_anchor * sin(angleRadians))
            perspective_camera_node.lookAt((0, 0, 0))
            return Task.cont

        if invoke_perspective_rotation:
            self.base.taskMgr.add(spinCameraTask, "SpinCameraTask")

    def setup_hud(self):
        # title = OnscreenText(text="Neural Box Modeling GT demo",
        #                      style=1, fg=(1, 1, 1, 1), pos=(-0.1, 0.1), scale=.07,
        #                      parent=base.a2dBottomRight, align=TextNode.ARight)
        # side_light_event = OnscreenText(text="1: Toggle Light from the front On/Off",
        #                                 style=1, fg=(1, 1, 1, 1), pos=(0.06, -0.08),
        #                                 align=TextNode.ALeft, scale=.05,
        #                                 parent=base.a2dTopLeft)
        # top_light_event = OnscreenText(text="2: Toggle Light from on top On/Off",
        #                                style=1, fg=(1, 1, 1, 1), pos=(0.06, -0.14),
        #                                align=TextNode.ALeft, scale=.05,
        #                                parent=base.a2dTopLeft)
        base = self.base
        front_title = OnscreenText(text="Front",
                                   style=1, fg=(1, 1, 1, 1), pos=(0.06, -0.08),
                                   align=TextNode.ALeft, scale=.05,
                                   parent=base.a2dTopLeft)
        top_title = OnscreenText(text="Top",
                                 style=1, fg=(1, 1, 1, 1), pos=(1.4, -0.08),
                                 align=TextNode.ALeft, scale=.05,
                                 parent=base.a2dTopLeft)
        left_title = OnscreenText(text="Left",
                                  style=1, fg=(1, 1, 1, 1), pos=(0.06, -1.08),
                                  align=TextNode.ALeft, scale=.05,
                                  parent=base.a2dTopLeft)
        perspective = OnscreenText(text="Perspective",
                                   style=1, fg=(1, 1, 1, 1), pos=(1.4, -1.08),
                                   align=TextNode.ALeft, scale=.05,
                                   parent=base.a2dTopLeft)

        self.hud.extend([front_title, top_title, left_title, perspective])

    def teardown_hud(self):
        for element in self.hud:
            element.destroy()

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
        # self.toggleLightsUp()

    def toggleLightsSide(self):
        self.is_light_side_on = not self.is_light_side_on

        if self.is_light_side_on:
            render.setLight(self.light_side)
            self.light_side.setPos(self.scene_graph_root, 1, -30, 0)
            self.light_side.lookAt(0, 0, 0)
        else:
            render.setLightOff(self.light_side)

    def toggleLightsUp(self):
        self.is_light_top_on = not self.is_light_top_on

        if self.is_light_top_on:
            render.setLight(self.light_top)
            self.light_top.setPos(self.scene_graph_root, 1, 0, 30)
            self.light_top.lookAt(0, 0, 0)
        else:
            render.setLightOff(self.light_top)

    def toggleWireframe(self):
        self.wireframe_mode += 1
        self.wireframe_mode %= 3

        if self.wireframe_mode == 0:
            render.setRenderModeFilledWireframe(wireframe_color=(1, 1, 0, 1))
        elif self.wireframe_mode == 1:
            render.setRenderModeWireframe()
        elif self.wireframe_mode == 2:
            render.setRenderModeFilled()

    def load_model(self, path):
        # Get the location of the 'py' file I'm running:
        mydir = os.path.dirname(os.path.abspath(sys.path[0]))

        # Convert that to panda's unix-style notation.
        mydir = Filename.fromOsSpecific(mydir).getFullpath()

        # Now load the model:
        model = loader.loadModel(mydir + path)
        # model.reparentTo(render)

        # model.setScale(100)
        # model.hprInterval(1.5, (360, 360, 360)).loop()

        return model

    def _dr_by_name(self, camera_name):
        if camera_name == 'top':
            return self.dr_top
        elif camera_name == 'left':
            return self.dr_left
        elif camera_name == 'front':
            return self.dr_front
        else:
            return self.dr_perspective

    def get_camera_image(self, camera_name='left', requested_format=None):
        """
        Returns the camera's image, which is of type uint8 and has values
        between 0 and 255.
        The 'requested_format' argument should specify in which order the
        components of the image must be. For example, valid format strings are
        "RGBA" and "BGRA". By default, Panda's internal format "BGRA" is used,
        in which case no data is copied over.
        """
        self.prepare_for_rendering()
        self.teardown_hud()
        self.base.graphicsEngine.renderFrame()
        dr = self._dr_by_name(camera_name)
        tex = dr.getScreenshot()
        num_components = tex.getNumComponents()
        if requested_format is None:
            data = tex.getRamImage()
        else:
            data = tex.getRamImageAs(requested_format)
            num_components = len(requested_format)
        image = np.frombuffer(data, np.uint8)  # use data.get_data() instead of data in python 2
        image.shape = (tex.getYSize(), tex.getXSize(), num_components)
        image = np.flipud(image)

        self.setup_hud()

        return image
