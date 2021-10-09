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
        return 202108311600
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
    def calculate_reward(self, curr:float, best:float, worst:float):
        raise NotImplementedError

    @abstractmethod
    def calculate_done(self, obs, reward):
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
        # print('on_init 1')
        context.stra_get_bars(
            stdCode='CFFEX.IF.HOT',
            period='m5',
            count=200,
            isMain=True
            )
        # print('on_init 2')
    
    def on_session_begin(self, context: CtaContext, curTDate: int):
        # print('on_session_begin')
        pass
    
    def on_calculate(self, context: CtaContext):
        # print('on_calculate 1')
        context.stra_log_text('%s%s'%(context.stra_get_date(), context.stra_get_time()))
        # print('on_calculate 2')
        # obs = 'obs'
        # reward = 1
        # done = False
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
        # print('on_calculate 3')

class TrainHFT(BaseHftStrategy, StateTransfer):
    @staticmethod
    def EngineType():
        return EngineType.ET_HFT

    def on_tick(self, context: HftContext, stdCode: str, newTick: dict):
        pass

class DemoCTA(TrainCTA):
    @staticmethod
    def Name():
        return __class__.__name__

    def calculate_obs(self, bars:WtKlineData):
        return bars.closes

    def calculate_reward(self, curr:float, best:float, worst:float):
        return 1

    def calculate_done(self, obs, reward):
        return False
        return True if np.random.randint(1, 100)==99 else False #是否结束

    def on_backtest_end(self, context: CtaContext):
        print('on_backtest_end')

    @staticmethod
    def TrainStartTime():
        return 201909100930
        return 202105201600

    @staticmethod
    def TrainEndTime():
        return 201912011500


class DemoHFT(TrainHFT):
    @staticmethod
    def Name():
        return __class__.__name__

    def calculate_obs(self, tick:dict):
        pass

    def calculate_reward(self, curr:float, best:float, worst:float):
        return 1

    def calculate_done(self, obs, reward):
        return False
        return True if np.random.randint(1, 100)==99 else False #是否结束

    def on_backtest_end(self, context: CtaContext):
        print('on_backtest_end')