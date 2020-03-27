from direct.showbase.DirectObject import DirectObject


class InteractiveWidget(DirectObject):

    def __init__(self, render_engine, animation=None):
        self.accept("1", render_engine.toggleLightsSide)
        self.accept("2", render_engine.toggleLightsUp)
        self.accept("w", render_engine.toggleWireframe)

        if not animation is None:
            self.accept("r", animation.reset)
            self.accept("s", animation.stop)
            self.accept("space", animation.toggle_active)
            self.accept("=", animation.increase_speed)
            self.accept("-", animation.decrease_speed)
