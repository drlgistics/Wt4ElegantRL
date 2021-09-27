import numpy as np
from wtpy.WtBtEngine import EngineType
from wtpy.StrategyDefs import BaseCtaStrategy, CtaContext, BaseHftStrategy, HftContext

class StateTransfer():
    @staticmethod
    def EngineType():
        raise NotImplementedError

    @staticmethod
    def Name():
        raise NotImplementedError

    @staticmethod
    def TrainStartTime():
        return 202105201600

    @staticmethod
    def TrainEndTime():
        return 202109091600

    def __init__(self) -> None:
        self.set_action(0)
        self.set_state(1, 1, False, {})
        print('StateTransfer')

    def get_action(self):
        return self.__action__

    def set_action(self, action):
        self.__action__ = action

    def get_state(self):
        done = True if np.random.randint(1, 100)==99 else False #是否结束
        return self.__obs__, self.__reward__, done, self.__info__

    def set_state(self, obs, reward, done, info):
        self.__obs__ = obs
        self.__reward__ = reward
        self.__done__ = done
        self.__info__ = info

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
        bars = context.stra_get_bars(
            stdCode='DCE.c.HOT',
            period='m1',
            count=60
            )
        # print(bars.closes)

class TrainHFT(BaseHftStrategy, StateTransfer):
    @staticmethod
    def EngineType():
        return EngineType.ET_HFT

class DemoCTA(TrainCTA):
    @staticmethod
    def Name():
        return __class__.__name__

class DemoHFT(TrainHFT):
    @staticmethod
    def Name():
        return __class__.__name__