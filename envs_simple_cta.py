from features import Indicator
from assessments import Assessment
from stoppers import SimpleStopper
from strategies import SimpleCTA
from envs import WtEvaluator, WtTrainer
from wtpy.StrategyDefs import CtaContext


class SimpleAssessment(Assessment): # 借鉴了neofinrl
    gamma = 0.99 
    scaling = 0.0005
    def reset(self):
        self.__assets__: list = [0]
        self.__reward__: list = []
        self.__done__: bool = False

    def calculate(self, context: CtaContext):
        if self.__done__:
            return 
        self.__assets__.append(context.stra_get_fund_data(0)) #账户实时的动态权益
        self.__reward__.append((self.__assets__[-1]-self.__assets__[-2])*self.scaling) # 以动态权益差分设计reward

        self.__done__ = False # 此处可以根据控制任务结束状态

    def finish(self):
        if self.__done__:
            return 
        gamma = 0
        for reward in self.__reward__:
            gamma = gamma*self.gamma + reward
        self.__reward__.append(gamma) # 在结束的时候把过程奖励做处理，作为整个训练的奖励
        self.__done__ = True

    @property
    def reward(self) -> float:
        return self.__reward__[-1]

    @property
    def done(self) -> float:
        return self.__done__

    @property
    def assets(self) -> float:
        return self.__assets__[-1]


class SimpleEvaluator(WtEvaluator):
    def __init__(self, time_start: int, time_end: int, id: int=1):
        # 角色：数据研究人员、强化学习研究人员、策略研究人员
        # 原则：每个角色的分工模拟交易机构做隔离


        # 特征工程组件, 滚动窗口=2，根据特征工程自动生成强化学习需要的observation
        # 特征工程的因子生成绝大多数情况下（舆情因子、周期因子）不是由env负责的，所以尽量使用特征工程组件而不要在env中定义因子
        # 特征工程的因子定义和生成，主要使用者是数据研究人员
        # 特征工程的因子后处理，主要使用者是强化学习研究人员
        feature: Indicator = Indicator(
            code='SHFE.rb.HOT', period=Indicator.M5, roll=3)  # 每一个特征工程必须指定一个主要标的

        # 按需添加其他标的
        feature.addSecurity(code='SHFE.hc.HOT')
        # feature.addSecurity(code='CFFEX.IC.HOT')

        # 分别使用1分钟、5分钟、15分钟线建立多周期因子
        for period in (feature.M5, feature.M15):
            feature.trange(period) # 波动率
            feature.macd(period) # 双均线强度
            feature.bollinger(period) # 标准差通道

        # 除上述特征，特征工程组件会自动加上 "开仓的最大浮盈、开仓的最大亏损、开仓的浮动盈亏、当前持仓数"4列，如果没有持仓则全部为0

        # 止盈止损组件，暂时是个摆设
        # 止盈止损组件的主要使用者是策略研究人员
        stopper: SimpleStopper = SimpleStopper()

        # 评估组件
        # 评估组件的主要使用者是强化学习研究人员定义reward
        assessment: SimpleAssessment = SimpleAssessment()
        super().__init__(
            strategy=SimpleCTA, # 策略只做跟交易模式相关的操作(如趋势策略、日内回转、配对交易、统计套利)，不参与特征生成和评估，主要使用者是策略研究人员
            stopper=stopper, # 特征计算
            feature=feature, # 评估计算
            assessment=assessment, 
            time_start=time_start, 
            time_end=time_end, 
            id=id
            )

class SimpleTrainer(SimpleEvaluator, WtTrainer):
    def __init__(self, time_start: int, time_end: int, id: int=1):
        super().__init__(time_start=time_start, time_end=time_end, id=id)



if __name__ == '__main__':
    env: WtTrainer = SimpleTrainer(
        time_start=201909100930,
        time_end=201912011500,
        )

    for i in range(1):  # 模拟训练10次
        obs = env.reset()
        done = False
        n = 0
        while not done:
            action = env.action_space.sample()  # 模拟智能体产生动作
            obs, reward, done, info = env.step(action)
            n += 1
            print('action:', action, 'obs:', obs, 'reward:', reward, 'done:', done)
        print('第%s次训练完成，执行%s步, 盈亏%s。' % (i+1, n, env.assets))
    env.close()
