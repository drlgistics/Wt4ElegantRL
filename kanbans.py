import talib as ta

class Kanban():
    M1 = 'm1'
    M3 = 'm3'
    M5 = 'm5'
    M15 = 'm15'
    M30 = 'm30'
    M60 = 'm60'
    D1 = 'd1'

    @property
    def security(self):
        return self._securities_

    def __init__(self, code:str, period:str, roll:int) -> None:
        self._securities_:dict = {}
        self._period_:dict = {}
        self._main_:str = None
        self._count_:dict = {}

        self.addSecurity(code=code)

    def addSecurity(self, code:str):
        # security:dict = self._securities_.get(code, {})
        # security[period] = count
        # self._securities_[code] = security
        pass
    
    def _calculate_(self, period:str, callback, **kwargs):
        print(kwargs)
        pass

class Indicator(Kanban):
    def macd(self, period:str, fastperiod:int=12, slowperiod:int=26, signalperiod:int=9):
        self._calculate_(period=period, callback=ta.MACD, fastperiod=12, slowperiod=26, signalperiod=9)