from abc import abstractmethod, ABC


class AbstractModifier(ABC):

    def __init__(self, mesh):
        self.mesh = mesh

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass

    def mesh(self):
        return self.mesh
