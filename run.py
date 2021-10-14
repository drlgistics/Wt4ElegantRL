from click import command, group
from features import Indicator
from rewards import SimpleReward
from stoppers import SimpleStopper
from strategies import SimpleCTADemo
from envs import EvaluatorWt, TrainWt

@group()
def run():
    pass

@command()
def debug():
    # 特征工程组件
    feature:Indicator = Indicator(code='CFFEX.IF.HOT', period=Indicator.M5, roll=3)
    feature.addSecurity(code='CFFEX.IH.HOT')
    feature.macd(feature.M5)
    feature.macd(feature.M1)

    # 止盈止损组件
    stopper:SimpleStopper = SimpleStopper()

    # 奖励组件
    reward:SimpleReward = SimpleReward()

    # 环境组装
    env:TrainWt = TrainWt(
        strategy=SimpleCTADemo,
        feature=feature,
        reward=reward,
        stopper=stopper,
        time_start=201909100930, 
        time_end=201912011500
        )
 
    for i in range(10): #模拟训练10次
        obs = env.reset()
        done = False
        action = 0
        while not done:
            action += 1 #模拟智能体产生动作
            obs, reward, done, info = env.step(action)
            print('obs%s'%obs, 'reward%s'%reward, done, info)
        print('第%s次训练完成'%i)
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