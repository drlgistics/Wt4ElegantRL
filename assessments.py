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


class SimpleAssessment(Assessment): # 借鉴了neofinrl
    gamma = 0.99 
    scaling = 0.0005
    def reset(self):
        self.__assets__: list = [0]
        self.__reward__: list = []
        self.__done__: bool = False

    def calculate(self, context: CtaContext):
        if self.__done__:
            return 
        self.__assets__.append(context.stra_get_fund_data(0)) #账户实时的动态权益
        self.__reward__.append((self.__assets__[-1]-self.__assets__[-2])*self.scaling) # 以动态权益差分设计reward

        self.__done__ = False # 此处可以根据控制任务结束状态

    def finish(self):
        if self.__done__:
            return 
        gamma = 0
        for reward in self.__reward__:
            gamma = gamma*self.gamma + reward
        self.__reward__.append(gamma) # 在结束的时候把过程奖励做处理，作为整个训练的奖励
        self.__done__ = True

    @property
    def reward(self) -> float:
        return self.__reward__[-1]

    @property
    def done(self) -> float:
        return self.__done__

    @property
    def assets(self) -> float:
        return self.__assets__[-1]
