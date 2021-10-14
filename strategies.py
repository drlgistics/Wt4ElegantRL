from numpy import ndarray
from gym.spaces import Box
from rewards import Reward
from features import Feature
from stoppers import Stopper
from abc import abstractmethod
from wtpy.WtBtEngine import EngineType
from wtpy.WtDataDefs import WtKlineData
from wtpy.StrategyDefs import BaseCtaStrategy, CtaContext, BaseHftStrategy, HftContext


class StateTransfer():
    @staticmethod
    @abstractmethod
    def Name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def EngineType() -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def Action(size: int) -> Box:
        raise NotImplementedError

    def __init__(self, feature: Feature, reward: Reward, stopper: Stopper):
        self._feature_: Feature = feature
        self._reward_: Reward = reward
        self._stopper_: Stopper = stopper
        print('StateTransfer')

    def getAction(self):
        return self.__action__

    def setAction(self, action):
        # if action is not None:
        #     print([v for i, v in enumerate(action)])
        self.__action__ = action

    def getState(self):
        return self.__obs__, self.__reward__, self.__done__, self.__info__

    def setState(self, obs: ndarray, reward: float, done: bool, info: dict):
        self.__obs__: ndarray = obs
        self.__reward__: float = reward
        self.__done__: bool = done
        self.__info__: dict = info

    @abstractmethod
    def calculate_action(self, action) -> int:
        raise NotImplementedError


class SimpleCTA(BaseCtaStrategy, StateTransfer):
    @staticmethod
    def Name() -> str:
        return __class__.__name__

    @staticmethod
    def EngineType() -> int:
        return EngineType.ET_CTA

    @staticmethod
    def Action(size: int) -> Box:
        return Box(low=-100, high=100, shape=(size, 1), dtype=int)

    def __init__(self, name: str, feature: Feature, reward: Reward, stopper: Stopper):
        super(BaseCtaStrategy, self).__init__(
            feature=feature, reward=reward, stopper=stopper)
        super().__init__(name)
        print('TrainCTA')

    def on_init(self, context: CtaContext):
        # print('on_init 1')
        self._feature_.subscribe(context)
        # print('on_init 2')

    def on_session_begin(self, context: CtaContext, curTDate: int):
        # print('on_session_begin')
        pass

    def on_backtest_end(self, context: CtaContext):
        # print('on_backtest_end')
        pass

    def on_calculate(self, context: CtaContext):
        # print('on_calculate 1')
        obs = self._feature_.calculate(context)
        reward, done = self._reward_.calculate(context=context)
        self.setState(obs, reward, done, {})
        # print('on_calculate 2')


class SimpleHFT(BaseHftStrategy, StateTransfer):
    @staticmethod
    def Name() -> str:
        return __class__.__name__

    @staticmethod
    def EngineType() -> int:
        return EngineType.ET_HFT

    def on_tick(self, context: HftContext, stdCode: str, newTick: dict):
        pass
