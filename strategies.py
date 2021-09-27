import numpy as np
from abc import abstractmethod
from wtpy.WtBtEngine import EngineType
from wtpy.WtDataDefs import WtKlineData
from wtpy.StrategyDefs import BaseCtaStrategy, CtaContext, BaseHftStrategy, HftContext

class StateTransfer():
    @staticmethod
    @abstractmethod
    def EngineType():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def Name():
        raise NotImplementedError

    @staticmethod
    def TrainStartTime():
        return 202105201600

    @staticmethod
    def TrainEndTime():
        return 202109091600

    def __init__(self):
        self.set_action(0)
        self.set_state(None, None, False, {})
        print('StateTransfer')

    def get_action(self):
        return self.__action__

    def set_action(self, action):
        self.__action__ = action

    def get_state(self):
        return self.__obs__, self.__reward__, self.__done__, self.__info__

    def set_state(self, obs, reward, done, info):
        self.__obs__ = obs
        self.__reward__ = reward
        self.__done__ = done
        self.__info__ = info

    @abstractmethod
    def calculate_obs(self, bars:WtKlineData):
        raise NotImplementedError

    @abstractmethod
    def calculate_reward(self, curr:int, best:int, worst:int):
        raise NotImplementedError

    @abstractmethod
    def calculate_done(self, obs, done):
        raise NotImplementedError

class TrainCTA(BaseCtaStrategy, StateTransfer):
    @staticmethod
    def EngineType():
        return EngineType.ET_CTA

    def __init__(self, name: str):
        super(BaseCtaStrategy, self).__init__()
        super().__init__(name)
        print('TrainCTA')

    def on_init(self, context: CtaContext):
        context.stra_get_bars(
            stdCode='DCE.c.HOT',
            period='m1',
            count=60,
            isMain=True
            )
    
    def on_calculate(self, context: CtaContext):
        obs = self.calculate_obs(
            context.stra_get_bars(
                stdCode='DCE.c.HOT',
                period='m1',
                count=60
                ))
        reward = self.calculate_reward(
            curr=context.stra_get_detail_profit(stdCode='DCE.c.HOT', usertag='', flag=0),
            best=context.stra_get_detail_profit(stdCode='DCE.c.HOT', usertag='', flag=1),
            worst=context.stra_get_detail_profit(stdCode='DCE.c.HOT', usertag='', flag=-1),
            )
        done = self.calculate_done(obs, reward)
        self.set_state(obs, reward, done, {})

class TrainHFT(BaseHftStrategy, StateTransfer):
    @staticmethod
    def EngineType():
        return EngineType.ET_HFT

class DemoCTA(TrainCTA):
    @staticmethod
    def Name():
        return __class__.__name__

    def calculate_obs(self, bars:WtKlineData):
        return bars.closes

    def calculate_reward(self, curr:int, best:int, worst:int):
        return 1

    def calculate_done(self, obs, done):
        return True if np.random.randint(1, 100)==99 else False #是否结束


class DemoHFT(TrainHFT):
    @staticmethod
    def Name():
        return __class__.__name__