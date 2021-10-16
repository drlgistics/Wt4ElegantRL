from abc import abstractmethod
from wtpy.StrategyDefs import CtaContext, HftContext


class Assessment():
    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def calculate(self, context: CtaContext):
        raise NotImplementedError

    @abstractmethod
    def finish(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def reward(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def done(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def assets(self) -> float:
        raise NotImplementedError
