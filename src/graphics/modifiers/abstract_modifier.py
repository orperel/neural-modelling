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

    @abstractmethod
    def affected_elements(self) -> dict:
        """
        :return:
        {
            'pre_modification': {
                'vertices': [ vid1, ...],
                'edges': [ eid1, ...],
                'faces': [ fid1, ...]
            },
            'post_modification': {
                'vertices': [ vid1, ...],
                'edges': [ eid1, ...],
                'faces': [ fid1, ...]
            }
        }
        """
        pass

    def mesh(self):
        return self.mesh
