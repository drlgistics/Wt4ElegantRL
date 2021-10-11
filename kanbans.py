import talib

class Kanban():
    def __init__(self, code:str, period:str, count:int, roll:int) -> None:
        self.__sub__:list = list()
        self.sub(code=code, period=period, count=count)
        
    def sub(self, code:str, period:str, count:int) -> None:
        self.__sub__.append((code, period, count))
    
    def add(self, period:str, cb, **kwargs) -> None:
        pass

class FactorsKanban(Kanban):
    pass

class TalibFactor():
    @staticmethod
    def SMA(timeperiod:int):
        pass