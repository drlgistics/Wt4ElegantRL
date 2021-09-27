from gym import Env
from wtpy.WtBtEngine import WtBtEngine
from strategies import StateTransfer, EngineType

class EnvWt(Env):
    def __init__(self, cls):
        self._iter_ = 0

        self.__cls__ = cls
        self._cls_:StateTransfer = None

        self._et_ = self.__cls__.EngineType()

        # 创建一个运行环境
        self._engine_:WtBtEngine = WtBtEngine(
            eType=self._et_,
            logCfg='./config/03research/log.json'
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

        # 创建一个策略并加入运行环境
        self._cls_:StateTransfer = self.__cls__(name='%s_%s'%(self.__cls__.Name(), self._iter_))

        # 设置策略的时候一定要安装钩子
        if self._et_ == EngineType.ET_CTA:
            self._engine_.set_cta_strategy(self._cls_, hook=True, slippage=1)#
        elif self._et_ == EngineType.ET_HFT:
            self._engine_.set_hft_strategy(self._cls_, hook=True)#
        else:
            raise AttributeError

        # 回测一定要异步运行
        self._engine_.run_backtest(bAsync=True)
        return self.step(0)[0]
    
    def step(self, action):
        assert self._iter_>0
        self._cls_.set_action(action)
        self._cb_step_()
        obs, reward, done, info = self._cls_.get_state()
        return obs, reward, done, info

    def close(self):
        if self._iter_>0:
            self._engine_.stop_backtest()
            self._engine_.clear_cache()

    def __del__(self):
        self._engine_.release_backtest()