from gym import Env
from gym.spaces import Box
from rewards import Reward
from features import Feature
from stoppers import Stopper
from wtpy.WtBtEngine import WtBtEngine
from strategies import StateTransfer, EngineType

#一个进程只能有一个env
class EvaluatorWt(Env):
    _log_:str = './config/03research/log_evaluator.json'
    _dump_:bool = True

    def __init__(self, strategy:StateTransfer, feature:Feature, reward:Reward, stopper:Stopper, time_start:int, time_end:int, id:int=1):
        self._id_:int = id
        self._iter_:int = 0
        self._run_:bool = False 

        self.__strategy__ = strategy
        self._et_ = self.__strategy__.EngineType()

        self.__feature__:Feature = feature
        self.observation_space:Box = self.__feature__.observation()
        self.action_space:Box = self.__strategy__.Action(self.__feature__.shape[0])
        
        self.__reward__:Reward = reward
        self.__stopper__:Stopper = stopper


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
        
        self._engine_.configBacktest(time_start, time_end)
        self._engine_.commitBTConfig()
        
    def reset(self):
        self.close()
        self._iter_ += 1

        # 创建一个策略并加入运行环境
        self._strategy_:StateTransfer = self.__strategy__(
            name=self._name_(),
            feature=self.__feature__,
            stopper=self.__stopper__,
            reward=self.__reward__,
            )

        # 设置策略的时候一定要安装钩子
        if self._et_ == EngineType.ET_CTA:
            self._engine_.set_cta_strategy(self._strategy_, slippage=1, hook=True, persistData=self._dump_)#
        elif self._et_ == EngineType.ET_HFT:
            self._engine_.set_hft_strategy(self._strategy_, hook=True)#
        else:
            raise AttributeError

        # 回测一定要异步运行
        self._engine_.run_backtest(bAsync=True, bNeedDump=self._dump_)
        self._run_ = True

        return self.step(0)[0]
    
    def step(self, action):
        assert self._iter_>0
        self._strategy_.setAction(action)
        finished = not self._cb_step_()
        obs, reward, done, info = self._strategy_.getState()
        if done or finished:
            done = True
            self.close()
        return obs, reward, done, info

    def close(self):
        if self._run_:
            self._engine_.stop_backtest()
            self._run_ = False

    def _name_(self):
        return '%s%s_%s%s'%(__class__.__name__, self._id_, self.__strategy__.Name(), self._iter_)

    def __del__(self):
        self._engine_.release_backtest()


class TrainWt(EvaluatorWt):
    _log_:str = './config/03research/log_train.json'
    _dump_:bool = False
    def _name_(self):
        return '%s%s_%s'%(__class__.__name__, self._id_, self.__strategy__.Name())