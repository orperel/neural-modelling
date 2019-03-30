from framework.event_handler import EventHandler


class EventDelegator:

    def __init__(self):
        self._handlers = []

    def __iadd__(self, handler: EventHandler):
        self._handlers.append(handler)
        return self

    def __isub__(self, handler: EventHandler):
        self._handlers.remove(handler)
        return self

    def fire(self, *args, **keywargs):
        for handler in self._handlers:
            handler(*args, **keywargs)
