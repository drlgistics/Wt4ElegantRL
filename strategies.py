from features import Feature
from stoppers import Stopper
from abc import abstractmethod
from assessments import Assessment
from wtpy.WtBtEngine import EngineType
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
    def Action(size: int) -> dict:
        raise NotImplementedError

    @staticmethod
    def setAction(self, action):
        raise NotImplementedError

    def __init__(self, feature: Feature, assessment: Assessment, stopper: Stopper):
        self._feature_: Feature = feature
        self._assessment_: Assessment = assessment
        self._stopper_: Stopper = stopper

        # print('StateTransfer')


class SimpleCTA(BaseCtaStrategy, StateTransfer):
    @staticmethod
    def Name() -> str:
        return __class__.__name__

    @staticmethod
    def EngineType() -> int:
        return EngineType.ET_CTA

    @staticmethod
    def Action(size: int) -> dict:
        return dict(low=-1, high=1, shape=(size, ), dtype=int)

    def setAction(self, action):
        # print('setAction 1')
        if action is not None:
            self._action_ = dict(zip(self._feature_.securities, action))
        # print('setAction 2')

    def __init__(self, name: str, feature: Feature, assessment: Assessment, stopper: Stopper):
        super(BaseCtaStrategy, self).__init__(
            feature=feature, assessment=assessment, stopper=stopper)
        super().__init__(name)
        self._action_: dict = {}
        # print('TrainCTA')

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
        # for code in tuple(self._action_.keys()):
        #     context.stra_set_position(stdCode=code, qty=self._action_.pop(code))
        #     print('stra_set_position %s'%code)

        # print('on_calculate 1')
        self._feature_.calculate(context=context)
        self._assessment_.calculate(context=context)
        # print('on_calculate 2')

    def on_tick(self, context: CtaContext, stdCode: str, newTick: dict):
        if stdCode not in self._action_:
            return
        # print('on_tick 1')
        context.stra_set_position(
            stdCode=stdCode, qty=self._action_.pop(stdCode))
        # print('on_tick 2')
        pass


# class SimpleHFT(BaseHftStrategy, StateTransfer):
#     @staticmethod
#     def Name() -> str:
#         return __class__.__name__

#     @staticmethod
#     def EngineType() -> int:
#         return EngineType.ET_HFT

#     def on_tick(self, context: HftContext, stdCode: str, newTick: dict):
#         pass
