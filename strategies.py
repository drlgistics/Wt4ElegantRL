import numpy as np
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
    def EngineType() -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def Name() -> str:
        raise NotImplementedError

    def __init__(self, feature: Feature, reward: Reward, stopper: Stopper):
        self._feature_: Feature = feature
        self._reward_: Reward = reward
        self._stopper_: Stopper = stopper

        self.set_action(0)
        self.set_state(None, None, False, {})
        print('StateTransfer')

    def get_action(self) -> int:
        return self.__action__

    def set_action(self, action):
        self.__action__: int = self.calculate_action(action)

    def get_state(self):
        return self.__obs__, self.__reward__, self.__done__, self.__info__

    def set_state(self, obs, reward: float, done: bool, info: dict):
        self.__obs__ = obs
        self.__reward__: float = reward
        self.__done__: bool = done
        self.__info__: dict = info

    @abstractmethod
    def calculate_action(self, action) -> int:
        raise NotImplementedError


class SimpleCTA(BaseCtaStrategy, StateTransfer):
    @staticmethod
    def EngineType() -> int:
        return EngineType.ET_CTA

    def __init__(self, name: str, feature: Feature, reward: Reward, stopper: Stopper):
        super(BaseCtaStrategy, self).__init__(
            feature=feature, reward=reward, stopper=stopper)
        super().__init__(name)
        print('TrainCTA')

    def on_init(self, context: CtaContext):
        # context.stra_log_text('on_init 1')
        self._feature_.subscribe(context)
        # context.stra_log_text('on_init 2')

    def on_session_begin(self, context: CtaContext, curTDate: int):
        # context.stra_log_text('on_session_begin')
        pass

    def on_backtest_end(self, context: CtaContext):
        # context.stra_log_text('on_backtest_end')
        pass

    def on_calculate(self, context: CtaContext):
        # context.stra_log_text('on_calculate 1')
        obs = self._feature_.calculate(context)
        reward, done = self._reward_.calculate(context)
        self.set_state(obs, reward, done, {})
        # context.stra_log_text('on_calculate 2')


class SimpleHFT(BaseHftStrategy, StateTransfer):
    @staticmethod
    def EngineType() -> int:
        return EngineType.ET_HFT

    def on_tick(self, context: HftContext, stdCode: str, newTick: dict):
        pass


class SimpleCTADemo(SimpleCTA):
    @staticmethod
    def Name() -> str:
        return __class__.__name__

    def calculate_action(self, action) -> int:
        return int(action)


class SimpleHFTDemo(SimpleHFT):
    @staticmethod
    def Name() -> str:
        return __class__.__name__
