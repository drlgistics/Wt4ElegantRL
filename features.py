import talib as ta
from wtpy.WtDataDefs import WtKlineData, WtHftData
from wtpy.StrategyDefs import CtaContext, HftContext

class Feature():
    M1 = 'm1'
    M3 = 'm3'
    M5 = 'm5'
    M15 = 'm15'
    M30 = 'm30'
    M60 = 'm60'
    D1 = 'd1'

    def __init__(self, code: str, period: str, roll: int) -> None:
        self._securities_: list = []
        self.addSecurity(code=code)

        self._main_: tuple = (code, period)
        self._subscribies_: dict = {}
        self._subscribe_(period=period, count=1)

        self._roll_:int = roll

    def addSecurity(self, code: str):
        if code not in self._securities_:
            self._securities_.append(code)

    def _subscribe_(self, period:str, count:int):
        self._subscribies_[period] = max(
            self._subscribies_.get(period, 0),
            count
            )

    def subscribe(self, context:CtaContext):
        '''
        根据看板订阅数据
        '''
        for code in self._securities_:
            for period, count in self._subscribies_.items():
                context.stra_get_bars(
                    stdCode=code,
                    period=period,
                    count=count,
                    isMain=(code==self._main_[0] and period==self._main_[1])
                )

    def _callback_(self, period: str, callback, **kwargs):
        pass

    def calculate(self, context:CtaContext):
        print(context.stra_get_date(), context.stra_get_time())


class Indicator(Feature):
    def macd(self, period: str, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
        def _macd_(data:WtKlineData, args):
            return ta.MACD(data.closes, **args)
            
        self._subscribe_(period=period, count=slowperiod+signalperiod)
        self._callback_(period=period, callback=_macd_,
                         fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
