from src.app import VisualizePreModifierEventHandler
from src.app.events.visualize_post_modifier import VisualizePostModifierEventHandler

class Animation:

    def __init__(self, engine, modifiers):
        self.engine = engine
        self.modifiers = modifiers

        self.pre_modifier_event_handler = VisualizePreModifierEventHandler(self.engine)
        self.post_modifier_event_handler = VisualizePostModifierEventHandler(self.engine)
        self.pre_modifier_event_handler.render_time = 0
        self.post_modifier_event_handler.render_time = 0
        self.render_time = 1.0

        self.is_active = True
        self.past_frames = []
        self.future_frames = []

    def start(self):
        self.reset()
        self.animate()

    def reset(self):
        self.past_frames = []
        self.future_frames = self.modifiers.copy()

    def _step(self):
        import time
        start = time.time()
        while time.time() - start < self.render_time:
            self.engine.base.task_mgr.step()

    def animate(self):
        number_of_animated_frames = 0
        for modifier in self.future_frames:
            if not self.is_active:
                break
            self.pre_modifier_event_handler.handle(modifier)
            self._step()
            modifier.execute()
            self.post_modifier_event_handler.handle(modifier)
            self._step()
            number_of_animated_frames += 1

        self.past_frames.extend(self.future_frames[:number_of_animated_frames])
        self.future_frames = self.future_frames[number_of_animated_frames:]

    def toggle_active(self):
        self.is_active = not self.is_active
        if self.is_active:
            self.animate()

    def stop(self):
        self.engine.stop_rendering_loop()

    def increase_speed(self):
        # self.engine.stop_rendering_loop()
        # if self.pre_modifier_event_handler.render_time >= 0.01:
        #     self.pre_modifier_event_handler.render_time /= 2
        # if self.post_modifier_event_handler.render_time >= 0.01:
        #     self.post_modifier_event_handler.render_time /= 2
        if self.render_time >= 0.01:
            self.render_time /= 2

    def decrease_speed(self):
        # self.engine.stop_rendering_loop()
        # if self.pre_modifier_event_handler.render_time < 10:
        #     self.pre_modifier_event_handler.render_time *= 2
        # if self.post_modifier_event_handler.render_time < 10:
        #     self.post_modifier_event_handler.render_time *= 2
        if self.render_time < 10:
            self.render_time *= 2
