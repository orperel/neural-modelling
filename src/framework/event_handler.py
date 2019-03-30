from abc import ABC, abstractmethod


class EventHandler(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def handle(self, *args, **keywargs):
        pass

    def __call__(self, *args, **kwargs):
        self.handle(*args, **kwargs)
