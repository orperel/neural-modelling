from direct.showbase.DirectObject import DirectObject

class InteractiveWidget(DirectObject):

    def __init__(self, render_engine):
        self.accept("1", render_engine.toggleLightsSide)
        self.accept("2", render_engine.toggleLightsUp)
