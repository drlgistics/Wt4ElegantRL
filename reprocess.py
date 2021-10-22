import numpy as np
import talib as ta


class REPROCESS():
    @staticmethod
    def n() -> int:  # 定义至少需要多少条数据才能计算
        return 0

    @staticmethod
    def calculate(data: np.ndarray) -> np.ndarray:  # 计算方法
        return np.clip(data, -1, 1)


class ZSCORE(REPROCESS):
    @staticmethod
    def n() -> int:
        return 60

    @staticmethod
    def calculate(data: np.ndarray) -> np.ndarray:
        return np.clip((data-ta.MA(data, __class__.n()))/ta.STDDEV(data, __class__.n()), -1, 1)


class ZFILTER(REPROCESS):
    '''
    https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py
    '''
    @staticmethod
    def n() -> int:
        return 60

    @staticmethod
    def calculate(data: np.ndarray) -> np.ndarray:
        # 1e-8
        return np.clip((data-ta.MA(data, __class__.n()))-(ta.STDDEV(data, __class__.n())+1e-5), -1, 1)
