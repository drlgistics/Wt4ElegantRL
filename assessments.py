from abc import abstractmethod
import numpy as np
from wtpy.StrategyDefs import CtaContext, HftContext


class Assessment():
    def __init__(self, init_assets=1000000):
        self._init_assets_ = init_assets
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
    def curr_assets(self) -> float:
        raise NotImplementedError

    @property
    def init_assets(self) -> float:
        return self._init_assets_


class SimpleAssessment(Assessment):  # 借鉴了neofinrl
    gamma = 0.99

    def reset(self):
        self.__assets__: list = [self._init_assets_]
        self.__reward__: list = [0]
        self.__done__: bool = False

    def calculate(self, context: CtaContext):
        if self.__done__:
            return
        self.__assets__.append(
            self._init_assets_+context.stra_get_fund_data(0))  # 账户实时的动态权益

        returns = np.round(self.__assets__[-1]/self.__assets__[-2]-1, 5)
        self.__reward__.append(
            returns if returns > 0 else returns*1.2,
            )  # 以动态权益差分设计reward

        self.__done__ = False  # 此处可以根据控制任务结束状态

    def finish(self):
        if self.__done__:
            return

        # returns = np.add(1, self.__reward__).cumprod()
        # np.subtract(returns, 1, out=returns)

        # gamma = 0
        # for reward in self.__reward__:
        #     gamma = gamma*self.gamma + reward

        # gamma = np.round(np.nanprod(np.array(self.__reward__)+1, axis=0)-1, 5)
        gamma = self.__assets__[-1]/self.__assets__[0]-1
        self.__reward__.append(
            gamma if gamma > 0 else gamma*1.2,
            )  # 在结束的时候把过程奖励做处理，作为整个训练的奖励
        self.__done__ = True

    @property
    def reward(self) -> float:
        return self.__reward__[-1]

    @property
    def rewards(self) -> float:
        return self.__reward__

    @property
    def done(self) -> float:
        return self.__done__

    @property
    def curr_assets(self) -> float:
        return self.__assets__[-1]
