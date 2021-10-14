from abc import abstractmethod
from wtpy.StrategyDefs import CtaContext, HftContext


class Reward():
    def __init__(self) -> None:
        self.reset()

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def calculate(self, context: CtaContext) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def finish(self) -> float:
        raise NotImplementedError


class SimpleReward(Reward):
    def reset(self):
        self.__reward__: list = []

    def calculate(self, context: CtaContext) -> tuple:
        self.__reward__.append(context.stra_get_date()
                               * 10000+context.stra_get_time())
        return self.__reward__[-1], False

    def finish(self) -> float:
        return self.__reward__[-1]
