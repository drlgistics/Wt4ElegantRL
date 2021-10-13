import talib as ta
from wtpy.WtDataDefs import WtKlineData

class Kanban():
    M1 = 'm1'
    M3 = 'm3'
    M5 = 'm5'
    M15 = 'm15'
    M30 = 'm30'
    M60 = 'm60'
    D1 = 'd1'

    @property
    def security(self) -> list:
        return self._securities_

    @property
    def subscribe(self) -> dict:
        return self._subscribies_

    def __init__(self, code: str, period: str, roll: int) -> None:
        self._securities_: list = []
        self.addSecurity(code=code)

        self._main_: str = period
        self._subscribies_: dict = {self._main_: 1}

    def addSecurity(self, code: str):
        if code not in self._securities_:
            self._securities_.append(code)

    def _subscribe_(self, period:str, count:int):
        self._subscribies_[period] = max(
            self._subscribies_.get(period, 0),
            count
            )


    def _calculate_(self, period: str, callback, **kwargs):
        print(kwargs)
        pass


class Indicator(Kanban):
    def macd(self, period: str, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
        def _macd_(data:WtKlineData, args):
            return ta.MACD(data.closes, **args)
            
        self._subscribe_(period=period, count=slowperiod+signalperiod)
        self._calculate_(period=period, callback=_macd_,
                         fastperiod=12, slowperiod=26, signalperiod=9)
