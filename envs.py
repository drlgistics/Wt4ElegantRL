from gym import Env
from gym.spaces import Box
from features import Feature
from stoppers import Stopper
from assessments import Assessment
from wtpy.apps import WtBtAnalyst
from wtpy.WtBtEngine import WtBtEngine
from strategies import StateTransfer, EngineType

# 一个进程只能有一个env


class WtEnv(Env):
    TRAINER = 1
    EVALUATOR = 2
    DEBUGGER = 3

    def __init__(self,
                 strategy: StateTransfer,
                 stopper: Stopper,
                 feature: Feature,
                 assessment: Assessment,
                 time_start: int,
                 time_end: int,
                 slippage: int = 0,
                 id: int = 1,
                 mode=1,
                 ):

        if mode == 3:  # 调试模式
            self._log_: str = './config/03research/log_debugger.json'
            self._dump_: bool = True
            self._mode_: str = 'WtDebugger'
        elif mode == 2:  # 评估模式
            self._log_: str = './config/03research/log_evaluator.json'
            self._dump_: bool = True
            self._mode_: str = 'WtEvaluator'
        else:  # 训练模式
            self._log_: str = './config/03research/log_trainer.json'
            self._dump_: bool = False
            self._mode_: str = 'WtTrainer'

        self._id_: int = id
        self._iter_: int = 0
        self._run_: bool = False

        self.__strategy__ = strategy
        self._et_ = self.__strategy__.EngineType()
        self.__stopper__: Stopper = stopper
        self.__slippage__: int = slippage

        self.__feature__: Feature = feature
        self.observation_space: Box = Box(**self.__feature__.observation)
        self.action_space: Box = Box(
            **self.__strategy__.Action(len(self.__feature__.securities)))

        self._assessment_: Assessment = assessment

        self.__time_start__ = time_start
        self.__time_end__ = time_end

    def _debug_(self):
        pass

    def __step__(self):
        finished = not self._cb_step_()
        if self._assessment_.done or finished:
            self._assessment_.finish()
            self._debug_()
            self.close()

    def close(self):
        if self._run_ and hasattr(self, '_engine_'):
            self._engine_.stop_backtest()
            self._run_ = False

    def reset(self):
        self.close()
        self._iter_ += 1

        if not hasattr(self, '_engine_'):
            # 创建一个运行环境
            self._engine_: WtBtEngine = WtBtEngine(
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

            self._engine_.configBacktest(
                self.__time_start__, self.__time_end__)
            self._engine_.commitBTConfig()

        # 重置奖励
        self._assessment_.reset()

        # 创建一个策略并加入运行环境
        self._strategy_: StateTransfer = self.__strategy__(
            name=self._name_(self._iter_),
            feature=self.__feature__,
            stopper=self.__stopper__,
            assessment=self._assessment_,
        )

        # 设置策略的时候一定要安装钩子
        if self._et_ == EngineType.ET_CTA:
            self._engine_.set_cta_strategy(
                self._strategy_, slippage=self.__slippage__, hook=True, persistData=self._dump_)
        elif self._et_ == EngineType.ET_HFT:
            self._engine_.set_hft_strategy(self._strategy_, hook=True)
        else:
            raise AttributeError

        # 回测一定要异步运行
        self._engine_.run_backtest(bAsync=True, bNeedDump=self._dump_)
        self._run_ = True

        self.__step__()
        return self.__feature__.obs

    def step(self, action):
        assert hasattr(self, '_engine_')
        self._strategy_.setAction(action)
        self._cb_step_()

        self.__step__()
        return self.__feature__.obs, self._assessment_.reward, self._assessment_.done, {}

    @property
    def assets(self):
        return self._assessment_.assets

    def analysis(self):
        for iter in range(1, self._iter_+1):
            name = self._name_(iter)
            analyst = WtBtAnalyst()
            analyst.add_strategy(name, folder="./outputs_bt/%s/" %
                                 name, init_capital=1000000, rf=0.02, annual_trading_days=240)
            try:
                analyst.run_new()
            except:
                analyst.run()

    def _name_(self, iter):
        return '%s%s_%s%s' % (self._mode_, self._id_, self.__strategy__.Name(), iter)

    def __del__(self):
        if hasattr(self, '_engine_'):
            self._engine_.release_backtest()
