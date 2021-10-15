from click import command, group
from features import Indicator
from assessments import SimpleAssessment
from stoppers import SimpleStopper
from strategies import SimpleCTA
from envs import EvaluatorWt, TrainWt


@group()
def run():
    pass


@command()
def debug():
    # 角色：数据研究人员、强化学习研究人员、策略研究人员
    # 原则：每个角色的分工模拟交易机构做隔离


    # 特征工程组件, 滚动窗口=2，根据特征工程自动生成强化学习需要的observation
    # 特征工程的因子生成绝大多数情况下（舆情因子、周期因子）不是由env负责的，所以尽量使用特征工程组件而不要在env中定义因子
    # 特征工程的因子定义和生成，主要使用者是数据研究人员
    # 特征工程的因子后处理，主要使用者是强化学习研究人员
    feature: Indicator = Indicator(
        code='CFFEX.IF.HOT', period=Indicator.M1, roll=2)  # 每一个特征工程必须指定一个主要标的

    # 按需添加其他标的
    feature.addSecurity(code='CFFEX.IH.HOT')
    feature.addSecurity(code='CFFEX.IC.HOT')

    # 分别使用1分钟、5分钟线建立多周期因子
    for period in (feature.M1, feature.M5):
        feature.atr(period)
        feature.macd(period)
        feature.bollinger(period)

    # 除上述特征，特征工程组件会自动加上 "开仓的浮动盈亏、开仓的最大浮盈、开仓的最大亏损、当前持仓数"4列，如果没有持仓则全部为0

    # 止盈止损组件，暂时是个摆设
    # 止盈止损组件的主要使用者是策略研究人员
    stopper: SimpleStopper = SimpleStopper()

    # 评估组件
    # 评估组件的主要使用者是强化学习研究人员
    assessment: SimpleAssessment = SimpleAssessment()

    # 环境组装，每一个进程只能有一个环境
    env: TrainWt = TrainWt(
        strategy=SimpleCTA,  # 策略只做跟交易模式相关的操作(如趋势策略、日内回转、配对交易、统计套利)，不参与特征生成和评估，主要使用者是策略研究人员
        feature=feature,  # 特征计算
        assessment=assessment,  # 评估计算
        stopper=stopper,
        time_start=201909100930,
        time_end=201912011500
    )

    for i in range(5000):  # 模拟训练10次
        obs = env.reset()
        done = False
        n = 0
        while not done:
            action = env.action_space.sample()  # 模拟智能体产生动作
            obs, reward, done, info = env.step(action)
            n += 1
            # print('action:', action, 'obs:', obs, 'reward:', reward, 'done:', done)
        print('第%s次训练完成，执行%s步。' % (i+1, n))
    env.close()

# @command()
# def train():
#     env = EvaluatorWt(cls=SimpleCTADemo, time_start=201909100930, time_end=201912011500)
#     for i in range(10): #模拟训练10次
#         print('第%s次训练'%i)
#         obs = env.reset()
#         done = False
#         action = 0
#         while not done:
#             action += 1 #模拟智能体产生动作
#             obs, reward, done, info = env.step(action)
#             # print('obs%s'%obs, 'reward%s'%reward, done, info)
#     env.close()


# run.add_command(train)
run.add_command(debug)

if __name__ == '__main__':
    run()
