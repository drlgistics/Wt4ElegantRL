from abc import abstractmethod
from wtpy.StrategyDefs import CtaContext, HftContext

class Reward():
    @abstractmethod
    def calculate(self, context:CtaContext) -> tuple:
        raise NotImplementedError

class SimpleReward(Reward):
    def calculate(self, context:CtaContext):
        return '%s%s'%(context.stra_get_date(), context.stra_get_time()), False