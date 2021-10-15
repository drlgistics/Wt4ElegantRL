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
        self.__data__: list = [0]
        self.__reward__: list = []
        self.__done__: bool = False

    def calculate(self, context: CtaContext):
        if self.__done__:
            return 
        self.__data__.append(context.stra_get_fund_data(0)) #账户实时的动态权益
        self.__reward__.append((self.__data__[-1]-self.__data__[-2])) # 以动态权益差分设计reward

        self.__done__ = False # 此处可以根据控制任务结束状态

    def finish(self):
        self.__reward__.append(sum(self.__reward__)) # 在结束的时候把过程奖励做处理，作为整个训练的奖励
        self.__done__ = True

    @property
    def reward(self) -> float:
        return self.__reward__[-1]

    @property
    def done(self) -> float:
        return self.__done__
