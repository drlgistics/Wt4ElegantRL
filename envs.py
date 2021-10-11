#from gym import Env
from wtpy.WtBtEngine import WtBtEngine
from strategies import StateTransfer, EngineType

class EvaluatorWt():#Env
    _log_:str = './config/03research/log_evaluator.json'

    def __init__(self, cls:StateTransfer, id:int=1):
        self._id_:int = id
        self._iter_:int = 0

        self.__cls__ = cls
        self._cls_:StateTransfer = None

        self._et_ = self.__cls__.EngineType()

        self._obs_ = None
        self._reward_:float = 0.
        self._done_:bool = False 
        self._info_:dict = {}

        self._run_:bool = False 

        # 创建一个运行环境
        self._engine_:WtBtEngine = WtBtEngine(
            eType=self._et_,
            logCfg=self._log_,
            )
        if self._et_ == EngineType.ET_CTA:
            self._engine_.init(
                './config/01commom/', 
                './config/03research/cta.json')
            self._cb_step_ = self._engine_.cta_step
        elif self._et_ == EngineType.ET_HFT:
            self._engine_.init(
                './config/01commom/', 
                './config/03research/hft.json')
            self._cb_step_ = self._engine_.hft_step
        else:
            raise AttributeError
        
        self._engine_.configBacktest(self.__cls__.TrainStartTime(), self.__cls__.TrainEndTime())
        self._engine_.commitBTConfig()
        
    def reset(self):
        self.close()
        self._iter_ += 1

        self._obs_ = None
        self._reward_:float = 0.
        self._done_:bool = False 
        self._info_:dict = {}

        # 创建一个策略并加入运行环境
        self._cls_:StateTransfer = self.__cls__(name=self._name_())

        # 设置策略的时候一定要安装钩子
        if self._et_ == EngineType.ET_CTA:
            self._engine_.set_cta_strategy(self._cls_, hook=True, slippage=1)#
        elif self._et_ == EngineType.ET_HFT:
            self._engine_.set_hft_strategy(self._cls_, hook=True)#
        else:
            raise AttributeError

        # 回测一定要异步运行
        self._engine_.run_backtest(bAsync=True)
        self._run_ = True

        return self.step(0)[0]
    
    def step(self, action):
        assert self._iter_>0
        self._cls_.set_action(action)
        if self._cb_step_():
            self._obs_, self._reward_, self._done_, self._info_ = self._cls_.get_state()
        else:
            self._done_ = True
        if self._done_:
            self.close()
        return self._obs_, self._reward_, self._done_, self._info_

    def close(self):
        if self._run_:
            self._engine_.stop_backtest()
            self._run_ = False

    def __del__(self):
        self._engine_.release_backtest()

    def _name_(self):
        return '%s%s_%s%s'%(__class__.__name__, self._id_, self.__cls__.Name(), self._iter_)


class TrainWt(EvaluatorWt):
    _log_:str = './config/03research/log_train.json'
    def _name_(self):
        return '%s%s_%s'%(__class__.__name__, self._id_, self.__cls__.Name())