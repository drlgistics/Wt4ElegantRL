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
        self._securities_:list = []
        self.addSecurity(code=code)

        self._period_:list = [period]
        self._main_:str = period
        self._count_:dict = {period:1}


    def addSecurity(self, code:str):
        if code not in self._securities_:
            self._securities_.append(code)
    
    def _calculate_(self, period:str, callback, **kwargs):
        print(kwargs)
        pass

class Indicator(Kanban):
    def macd(self, period:str, fastperiod:int=12, slowperiod:int=26, signalperiod:int=9):
        self._calculate_(period=period, callback=ta.MACD, fastperiod=12, slowperiod=26, signalperiod=9)