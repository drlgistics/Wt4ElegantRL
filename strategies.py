import numpy as np
from kanbans import Kanban
from rewards import Reward
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

    def __init__(self, kanban:Kanban, reward:Reward, stopper:Stopper):
        self._kanban_:Kanban = kanban
        self._reward_:Reward = reward
        self._stopper_:Stopper = stopper

        print(enumerate(self._kanban_.security))

        # self.set_action(0)
        # self.set_state(None, None, False, {})
        print('StateTransfer')

    def get_action(self) -> int:
        return self.__action__

    def set_action(self, action):
        self.__action__:int = self.calculate_action(action)

    def get_state(self):
        return self.__obs__, self.__reward__, self.__done__, self.__info__

    def set_state(self, obs, reward:float, done:bool, info:dict):
        self.__obs__ = obs
        self.__reward__:float = reward
        self.__done__:bool = done
        self.__info__:dict = info

    @abstractmethod
    def calculate_action(self, action) -> int:
        raise NotImplementedError

    @abstractmethod
    def calculate_obs(self, bars:WtKlineData):
        raise NotImplementedError

    @abstractmethod
    def calculate_reward(self, curr:float, best:float, worst:float) -> float:
        raise NotImplementedError

    @abstractmethod
    def calculate_done(self, obs, reward) -> bool:
        raise NotImplementedError

class SimpleCTA(BaseCtaStrategy, StateTransfer):
    @staticmethod
    def EngineType() -> int:
        return EngineType.ET_CTA

    def __init__(self, name: str, kanban:Kanban, reward:Reward, stopper:Stopper):
        super(BaseCtaStrategy, self).__init__(kanban=kanban, reward=reward, stopper=stopper)
        super().__init__(name)
        print('TrainCTA')

    def on_init(self, context: CtaContext):
        # CtaContext.stra_log_text('on_init 1')
        context.stra_get_bars(
            stdCode='CFFEX.IF.HOT',
            period='m5',
            count=200,
            isMain=True
            )
        # CtaContext.stra_log_text('on_init 2')
    
    def on_session_begin(self, context: CtaContext, curTDate: int):
        # CtaContext.stra_log_text('on_session_begin')
        pass

    def on_backtest_end(self, context: CtaContext):
        # CtaContext.stra_log_text('on_backtest_end')
        pass
    
    def on_calculate(self, context: CtaContext):
        # CtaContext.stra_log_text('on_calculate 1')
        obs = self.calculate_obs(
            bars=context.stra_get_bars(
                stdCode='CFFEX.IF.HOT',
                period='m5',
                count=200
                ))
        reward = self.calculate_reward(
            curr=context.stra_get_detail_profit(stdCode='CFFEX.IF.HOT', usertag='', flag=0),
            best=context.stra_get_detail_profit(stdCode='CFFEX.IF.HOT', usertag='', flag=1),
            worst=context.stra_get_detail_profit(stdCode='CFFEX.IF.HOT', usertag='', flag=-1),
            )
        done = self.calculate_done(obs=obs, reward=reward)
        self.set_state(obs, reward, done, {})
        # CtaContext.stra_log_text('on_calculate 2')

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

    def calculate_obs(self, bars:WtKlineData):
        return bars.closes

    def calculate_reward(self, curr:float, best:float, worst:float) -> float:
        return 1

    def calculate_done(self, obs, reward) -> bool:
        return False
        return True if np.random.randint(1, 100)==99 else False #是否结束


class SimpleHFTDemo(SimpleHFT):
    @staticmethod
    def Name() -> str:
        return __class__.__name__

    def calculate_obs(self, tick:dict):
        pass

    def calculate_reward(self, curr:float, best:float, worst:float) -> float:
        return 1

    def calculate_done(self, obs, reward) -> bool:
        return False
        return True if np.random.randint(1, 100)==99 else False #是否结束