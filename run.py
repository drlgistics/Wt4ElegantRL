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
    # 特征工程组件
    feature:Indicator = Indicator(code='CFFEX.IF.HOT', period=Indicator.M5, roll=3) # 每一个特征工程必须指定一个主要标的
    
    # 按需添加其他合约
    feature.addSecurity(code='CFFEX.IH.HOT') 
    feature.addSecurity(code='CFFEX.IC.HOT')
    
    # 使用5分钟线建立特征
    feature.macd(feature.M5)
    
    # 使用1分钟线建立特征
    feature.macd(feature.M1)

    # 止盈止损组件，暂时是个摆设
    stopper:SimpleStopper = SimpleStopper()

    # 评估组件
    assessment:SimpleAssessment = SimpleAssessment()

    # 环境组装，每一个进程只能有一个环境
    env:TrainWt = TrainWt(
        strategy=SimpleCTA,  # 策略只做跟交易模式相关的操作(如趋势策略、日内回转、配对交易、统计套利)，不参与特征生成和评估
        feature=feature, # 特征计算
        assessment=assessment, # 评估计算
        stopper=stopper,
        time_start=201909100930, 
        time_end=201912011500
        )

    print(env.observation_space.sample())
    print(env.action_space.sample())
 
    for i in range(1): #模拟训练10次
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample() #模拟智能体产生动作
            obs, reward, done, info = env.step(action)
            print(action, obs, reward, done, info)
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