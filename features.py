import numpy as np
import talib as ta
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
        self.__shape__: tuple = tuple()
        self._roll_: int = int(roll)

        self.__cb__: dict = {}

        self.__obs__: dict = {}
        self.__time__: int = 0

        self.__securities__: list = []
        self.addSecurity(code=code)

        self.__main__: tuple = (code, period)
        self.__subscribies__: dict = {}
        self._subscribe_(period=period, count=1)

    @property
    def securities(self):
        return self.__securities__

    def addSecurity(self, code: str):
        if self.__shape__ or code in self.__securities__:
            return
        self.__securities__.append(code)

    def _subscribe_(self, period: str, count: int):
        self.__subscribies__[period] = max(
            self.__subscribies__.get(period, 0),
            count+self._roll_
        )

    def subscribe(self, context: CtaContext):
        '''
        根据特征需求订阅数据
        '''
        for code in self.__securities__:
            for period, count in self.__subscribies__.items():
                context.stra_get_bars(
                    stdCode=code,
                    period=period,
                    count=count,
                    isMain=(code == self.__main__[0]
                            and period == self.__main__[1])
                )

    def _callback_(self, space: int, period: str, callback, **kwargs):
        if self.__shape__ or space < 1:
            return
        if period not in self.__cb__:
            self.__cb__[period] = {}
        self.__cb__[period][callback.__name__] = (space, callback, kwargs)

    @property
    def observation(self) -> dict:
        '''
        根据特征需求生成observation
        '''
        self.__shape__ = (
            len(self.securities),
            sum(c[0] for v in self.__cb__.values()
                for c in v.values())*self._roll_+4
        )
        return dict(low=-np.inf, high=np.inf, shape=self.__shape__, dtype=float)

    def calculate(self, context: CtaContext):
        self.__time__ = context.stra_get_date()*10000+context.stra_get_time()
        if self.__time__ not in self.__obs__:
            obs = np.full(shape=self.__shape__, fill_value=np.nan, dtype=float)
            for i, code in enumerate(self.securities):  # 处理每一个标的
                n = 0
                for period, v in self.__cb__.items():  # 处理每一个周期
                    for space, callback, args in v.values():  # 处理每一个特征
                        features = callback(
                            context=context, code=code, period=period, args=args)  # 通过回调函数计算特征
                        if space == 1:
                            features = (features, )
                        for feature in features:  # 处理每一个返回值
                            obs[i][n:n+self._roll_] = feature[-self._roll_:]
                            n += self._roll_
            self.__obs__[self.__time__] = obs

        # 标的浮动盈亏
        self.__obs__[self.__time__][:, -4] = tuple(
            context.stra_get_detail_profit(stdCode=code, usertag='', flag=0) for code in self.securities)
        # 标的最大浮盈
        self.__obs__[self.__time__][:, -3] = tuple(
            context.stra_get_detail_profit(stdCode=code, usertag='', flag=1) for code in self.securities)
        # 标的最大亏损
        self.__obs__[self.__time__][:, -2] = tuple(
            context.stra_get_detail_profit(stdCode=code, usertag='', flag=-1) for code in self.securities)
        # 持仓数
        self.__obs__[self.__time__][:, -1] = tuple(
            context.stra_get_position(stdCode=code) for code in self.securities)

    @property
    def obs(self):
        return self.__obs__.get(self.__time__)


class Indicator(Feature):
    def atr(self, period: str, timeperiod=14):
        def atr(context: CtaContext, code: str, period: str, args: dict):
            bars = context.stra_get_bars(
                stdCode=code, period=period, count=self.__subscribies__[period])
            return ta.ATR(bars.highs, bars.lows, bars.closes, **args)

        self._subscribe_(period=period, count=timeperiod)
        self._callback_(space=1, period=period, callback=atr,
                        timeperiod=timeperiod)

    def macd(self, period: str, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
        def macd(context: CtaContext, code: str, period: str, args: dict):
            return ta.MACD(context.stra_get_bars(stdCode=code, period=period, count=self.__subscribies__[period]).closes, **args)

        self._subscribe_(period=period, count=slowperiod+signalperiod)
        self._callback_(space=3, period=period, callback=macd,
                        fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

    def bollinger(self, period: str, timeperiod=5, nbdevup=2, nbdevdn=2):
        def bollinger(context: CtaContext, code: str, period: str, args: dict):
            return ta.BBANDS(context.stra_get_bars(stdCode=code, period=period, count=self.__subscribies__[period]).closes, **args)

        self._subscribe_(period=period, count=timeperiod)
        self._callback_(space=3, period=period, callback=bollinger,
                        timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
