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


class SimpleAssessment(Assessment):
    def reset(self):
        self.__reward__: list = []
        self.__done__: bool = False

    def calculate(self, context: CtaContext):
        self.__reward__.append(context.stra_get_date()
                               * 10000+context.stra_get_time())
        self.__done__ = False

    def finish(self):
        self.__done__ = True

    @property
    def reward(self) -> float:
        return self.__reward__[-1]

    @property
    def done(self) -> float:
        return self.__done__
