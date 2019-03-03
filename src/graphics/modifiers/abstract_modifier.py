from abc import abstractmethod, ABC


class AbstractModifier(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass
